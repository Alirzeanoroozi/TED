"""Build the MMseqs2 lookup DB over TED's own sequence clusters (db/mmseqs/).

TED clustered its 324,389,697 TED-100 domain sequences with
    mmseqs easy-linclust --min-seq-id 0.5 -c 0.9 --cov-mode 5
into 120,748,700 clusters and then gave clusters the CATH labels of their members. A new domain that aligns
to a cluster representative with >= 50% identity and length ratio >= 0.9 would have joined that cluster, so
the lookup DB holds the 120.7M representatives, each with its cluster's label summary in the header:

    >REP_TED_ID TOP_LABEL LEVEL n=MEMBERS nlab=LABELLED share=FRACTION own=REP_LABEL:METHOD

Steps (each skipped if its output exists):
  1 clusters.sorted.tsv   ted_324m_seq_clustering.cathlabels.tsv.gz sorted by representative
  2 rep_summary.tsv       one line per cluster with the label summary
  3 ted100_reps.fasta     representative sequences read out of Foldseek's teddb. teddb stores the 365M TED
                          domain sequences as plain 'SEQ\\n\\0' records in nearly the row order of
                          ted_365m.domain_summary, so the two files are walked together and synchronised by
                          md5 (record md5 == md5_domain of the TED id it is assigned to). teddb.index/.lookup
                          are therefore not needed.
  4 ted100               mmseqs createdb
The CPU index (createindex) and GPU padded DB (makepaddedseqdb) are made by build_lookup_dbs.sbatch.
"""
import hashlib
import itertools
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLU = Path(os.environ.get("TED_CLU", ROOT / "data/ted/ted_324m_seq_clustering.cathlabels.tsv.gz"))
TEDDB = Path(os.environ.get("TED_TEDDB", ROOT / "data/foldseek_teddb/teddb"))
SUMMARY = Path(os.environ.get("TED_SUMMARY", ROOT / "data/ted/ted_365m.domain_summary.cath.globularity.taxid.tsv.gz"))
OUT = Path(os.environ.get("TED_MMSEQS_OUT", ROOT / "db/mmseqs"))
TMP = Path(os.environ.get("TMPDIR", OUT / "tmp"))
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
SORTMEM = os.environ.get("SORT_MEM", "40G")
MMSEQS = ROOT / "tools/bin/mmseqs"
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:7.0f}s] {msg}", flush=True)


def sh(cmd):
    log(cmd)
    subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=True, env={**os.environ, "LC_ALL": "C"})


SORT = f"sort -t $'\\t' -S {SORTMEM} --parallel={THREADS} -T {TMP}"
LEVEL = {"foldseek-H": "H", "foldseek-T": "T", "foldclass": "T", "-": "-"}

# 1. sort cluster table by representative
sorted_clu = OUT / "clusters.sorted.tsv"
if not sorted_clu.exists():
    sh(f"zcat {CLU} | {SORT} -k1,1 > {sorted_clu}.part && mv {sorted_clu}.part {sorted_clu}")

# 2. per-cluster label summary
summary = OUT / "rep_summary.tsv"
if not summary.exists():
    log("summarising clusters")
    n_clu = 0
    with open(sorted_clu) as fh, open(str(summary) + ".part", "w") as fo:
        fo.write("#rep\tn_members\tn_labelled\ttop_label\ttop_level\ttop_share\town_label\town_method\tn_labels\n")
        cur, members = None, []

        def flush(rep, members):
            counts = Counter((lab, LEVEL.get(m, "-")) for _, lab, m in members if lab != "-")
            nlab = sum(counts.values())
            own = next(((lab, m) for mem, lab, m in members if mem == rep), ("-", "-"))
            if counts:
                (top, lev), c = max(counts.items(), key=lambda kv: (kv[1], kv[0][1] == "H"))
                share = c / nlab
            else:
                top, lev, share = "-", "-", 0.0
            fo.write(f"{rep}\t{len(members)}\t{nlab}\t{top}\t{lev}\t{share:.3f}\t{own[0]}\t{own[1]}\t{len(counts)}\n")

        for line in fh:
            rep, mem, lab, meth = line.rstrip("\n").split("\t")
            if rep != cur:
                if cur is not None:
                    flush(cur, members)
                    n_clu += 1
                    if n_clu % 10_000_000 == 0:
                        log(f"{n_clu:,} clusters")
                cur, members = rep, []
            members.append((mem, lab, meth))
        if cur is not None:
            flush(cur, members)
            n_clu += 1
    os.rename(str(summary) + ".part", summary)
    log(f"{n_clu:,} clusters summarised")

# 3. FASTA of representatives: walk teddb records and summary rows together, verify md5, merge with reps
fasta = OUT / "ted100_reps.fasta"


def teddb_records(path, block=1 << 26):
    with open(path, "rb") as fh:
        rest = b""
        while True:
            buf = fh.read(block)
            if not buf:
                break
            parts = (rest + buf).split(b"\x00")
            rest = parts.pop()
            for p in parts:
                yield p.rstrip(b"\n")
        if rest.strip(b"\n"):
            yield rest.rstrip(b"\n")


def summary_rows(path):
    p = subprocess.Popen(f"zcat {path} | cut -f1,2", shell=True, stdout=subprocess.PIPE, bufsize=1 << 24)
    for line in p.stdout:
        tid, m = line.rstrip(b"\n").split(b"\t")
        yield tid.decode(), m.decode()
    if p.wait() != 0:
        raise RuntimeError("reading the summary failed")


def rep_rows(path):
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                f = line.rstrip("\n").split("\t")
                yield f[0], f"{f[0]} {f[3]} {f[4]} n={f[1]} nlab={f[2]} share={f[5]} own={f[6]}:{f[7]}"


if not fasta.exists():
    # teddb holds the same 364.8M domains as the summary, in almost the same order: proteins with 6-character
    # accessions are sorted differently relative to 10-character accessions sharing the prefix (A0A009 vs
    # A0A009F1M9), ~65 small displacements per 20M rows. So the two files are walked together and re-synchronised
    # by md5: a summary row that is passed over is parked in `pending` until a record with its md5 turns up.
    # Every sequence written is therefore one whose md5 equals the TED id's published md5_domain.
    import collections
    LOOKAHEAD = int(os.environ.get("TED_LOOKAHEAD", "200000"))
    log("loading representative headers")
    reps = dict(rep_rows(summary))
    n_rep = len(reps)
    log(f"{n_rep:,} representatives; writing FASTA (md5-synchronised walk over teddb + summary)")
    S = summary_rows(SUMMARY)
    sbuf = collections.deque()
    pending, orphans = {}, {}
    n = written = resync = via_pending = 0

    def fill(k):
        while len(sbuf) < k:
            x = next(S, None)
            if x is None:
                return
            sbuf.append(x)

    with open(str(fasta) + ".part", "w") as fo:
        def emit(tid, seq):
            global written
            hdr = reps.pop(tid, None)
            if hdr is not None:
                fo.write(f">{hdr}\n{seq.decode()}\n")
                written += 1

        for seq in teddb_records(TEDDB):
            n += 1
            mt = hashlib.md5(seq).hexdigest()
            fill(1)
            if sbuf and sbuf[0][1] == mt:
                emit(sbuf.popleft()[0], seq)
            elif mt in pending:
                tids = pending[mt]
                emit(tids.pop(), seq)
                via_pending += 1
                if not tids:
                    del pending[mt]
            else:
                fill(LOOKAHEAD)
                j = next((i for i, (_, m) in enumerate(sbuf) if m == mt), None)
                if j is None:
                    orphans.setdefault(mt, []).append(seq)      # its summary row is further away; settle at the end
                else:
                    resync += 1
                    for _ in range(j):
                        tid, m = sbuf.popleft()
                        pending.setdefault(m, []).append(tid)
                    emit(sbuf.popleft()[0], seq)
            if n % 20_000_000 == 0:
                log(f"{n:,} domains read, {written:,} representatives written, {resync} re-syncs, "
                    f"{via_pending} matched from pending, {sum(map(len, pending.values()))} pending, "
                    f"{sum(map(len, orphans.values()))} orphans")
        for tid, m in itertools.chain(sbuf, S):                 # summary rows never reached
            pending.setdefault(m, []).append(tid)
        settled = 0
        for m, seqs in orphans.items():
            tids = pending.get(m, [])
            while seqs and tids:
                emit(tids.pop(), seqs.pop())
                settled += 1
        left_recs = sum(map(len, orphans.values()))
    log(f"total {n:,} teddb records; {written:,}/{n_rep:,} representatives written; {resync} re-syncs, "
        f"{via_pending} from pending, {settled} settled at the end; unmatched: {left_recs} records, "
        f"{sum(len(v) for v in pending.values())} summary rows")
    if written < 0.9999 * n_rep:
        sys.exit(f"only {written:,}/{n_rep:,} representatives written; inputs do not line up")
    os.rename(str(fasta) + ".part", fasta)

# 4. MMseqs2 sequence DB
db = OUT / "ted100"
if not Path(str(db) + ".dbtype").exists():
    sh(f"{MMSEQS} createdb {fasta} {db} --threads {THREADS} -v 2")
log("done")
