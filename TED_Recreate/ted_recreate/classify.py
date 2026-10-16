"""CATH labels for domains, following how TED assigned them.

TED labelled its 324M TED-100 domains in three steps (Lau et al. 2024, Supp. Methods):
  1. Foldseek structure search against CATH 4.3 representatives
       H (superfamily): E < 0.019,    TM-score >  0.56, coverage >= 0.367
       T (fold):        E < 0.108662, TM-score >= 0.42, coverage >= 0.786
  2. where Foldseek gave nothing: the nearest CATH domain by Foldclass embedding (Merizo-search), accepted as a
     T-level label if TM-align > 0.5, normalised by the TED domain
  3. MMseqs2 sequence clusters (>= 50% identity, length ratio >= 0.9): a cluster takes the labels of its members

For a new domain the same three sources are tried in this order, and the first that gives a label decides:
  exact      the domain's sequence is in TED (md5 match)         -> TED's published label
  structure  steps 1 and 2, run here on the domain's structure   -> needs a PDB of the domain
  sequence   the domain would join one of TED's clusters         -> that cluster's label (the "MMseqs server")
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .chop import parse_chopping, work_directory
from .structure import md5, read_pdb_ca

TIERS = ("exact", "structure", "sequence")

# TED's Foldseek search settings and thresholds (Supp. Methods, "Domain classification using Foldseek")
FS_SEARCH = ("-e", 0.108662, "-c", 0.366757, "--cov-mode", 5, "-s", 10)
H_EVALUE, H_TM, H_COV = 0.019, 0.56, 0.367
T_EVALUE, T_TM, T_COV = 0.108662, 0.42, 0.786
FOLDCLASS_TM = 0.5
CLUSTER_IDENTITY, CLUSTER_LENGTH_RATIO = 0.5, 0.9       # TED's MMseqs2 clustering rule

FS_COLUMNS = "query,target,evalue,bits,prob,qtmscore,ttmscore,fident,qlen,tlen"
MM_COLUMNS = "query,target,theader,pident,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,qcov,tcov,evalue,bits"


def _run(*cmd):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} {cmd[1]} failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")


def _read_hits(path: Path, columns: str, n_text: int) -> list[dict]:
    """Rows of a Foldseek / MMseqs2 result table as dicts; all but the first `n_text` columns are numbers."""
    cols = columns.split(",")
    hits = []
    for line in path.read_text().splitlines():
        f = line.split("\t")
        hits.append({**dict(zip(cols[:n_text], f)), **{k: float(v) for k, v in zip(cols[n_text:], f[n_text:])}})
    return hits


def _read_map(path: Path, value_column: int) -> dict[str, str]:
    """{first column: chosen column} of a tab-separated file with a header line."""
    with open(path) as fh:
        next(fh)
        return {f[0]: f[value_column] for f in (line.rstrip("\n").split("\t") for line in fh)}


def truncate(label: str, level: str) -> str:
    """'3.40.50.300', 'T' -> '3.40.50'"""
    return ".".join(label.split(".")[:4 if level == "H" else 3])


# --- exact: the domain's sequence is one of TED's 365M domains -----------------------------------------------------
class TedMd5Index:
    """TED's published label for each of its domain sequences, looked up by md5 (scripts/build_ted_md5_index.py).
    The arrays are memory-mapped, so opening the index loads nothing."""

    LEVELS = ("-", "T", "H")

    def __init__(self):
        d = config.TED_MD5_INDEX
        if not (d / "md5_hi.npy").exists():
            raise FileNotFoundError(f"TED md5 index not built yet ({d}); run scripts/build_lookup_dbs.sbatch")
        self.hi, self.lo, self.label, self.level, self.method, self.ted_row, self.id_offset = (
            np.load(d / f"{name}.npy", mmap_mode="r")
            for name in ("md5_hi", "md5_lo", "label", "level", "method", "ted_row", "ted_ids.offsets"))
        meta = json.loads((d / "labels.json").read_text())
        self.labels, self.methods = meta["labels"], meta["methods"]
        self.ids = open(d / "ted_ids.txt", "rb")

    def _ted_id(self, row: int) -> str:
        start, end = int(self.id_offset[row]), int(self.id_offset[row + 1])
        self.ids.seek(start)
        return self.ids.read(end - start).decode().rstrip("\n")

    def lookup(self, seq: str, max_ids: int = 20) -> list[dict]:
        """Every TED domain with exactly this sequence (TED ids are given for the first `max_ids`)."""
        h = md5(seq)
        hi, lo = np.uint64(int(h[:16], 16)), np.uint64(int(h[16:], 16))
        found = []
        for i in range(int(np.searchsorted(self.hi, hi, side="left")), int(np.searchsorted(self.hi, hi, side="right"))):
            if self.lo[i] == lo:
                found.append({"ted_id": self._ted_id(int(self.ted_row[i])) if len(found) < max_ids else None,
                              "cath_label": self.labels[int(self.label[i])],
                              "level": self.LEVELS[int(self.level[i])],
                              "method": self.methods[int(self.method[i])]})
        return found


# --- structure: Foldseek with TED's thresholds, then Foldclass + TM-align ------------------------------------------
def foldseek_search(pdb_dir: Path, workdir: Path, threads: int = 8) -> dict[str, list[dict]]:
    """Search every PDB in `pdb_dir` against CATH 4.3 S40 with TED's settings. Returns {query: hits}.

    The paper does not say which Foldseek columns its "TM-score" and "coverage" are. On 387 TED-100 domains,
    TM = max(qtmscore, ttmscore) with coverage = shorter length / longer length (cov-mode 5, as in TED's search)
    reproduces TED's Foldseek outcome best: 375/387, against 368 for alntmscore and 366 for qtmscore."""
    out = workdir / "foldseek.m8"
    _run(config.FOLDSEEK, "easy-search", pdb_dir, config.CATH_FOLDSEEK_DB, out, workdir / "fs_tmp", *FS_SEARCH,
         "--max-seqs", 1000, "--threads", threads, "--format-output", FS_COLUMNS)
    hits = {}
    for h in _read_hits(out, FS_COLUMNS, 2):
        h["query"], h["target"] = h["query"].removesuffix(".pdb"), h["target"].removesuffix(".pdb")
        h["tm"] = max(h["qtmscore"], h["ttmscore"])
        h["coverage"] = min(h["qlen"], h["tlen"]) / max(h["qlen"], h["tlen"])
        hits.setdefault(h["query"], []).append(h)
    return hits


def ted_foldseek_rule(hits: list[dict], cath_labels: dict) -> dict | None:
    """TED's thresholds on one query's Foldseek hits: the best hit passing the H rule, else the T rule, else None."""
    rules = (("H", lambda h: h["evalue"] < H_EVALUE and h["tm"] > H_TM and h["coverage"] >= H_COV),
             ("T", lambda h: h["evalue"] < T_EVALUE and h["tm"] >= T_TM and h["coverage"] >= T_COV))
    hits = sorted(hits, key=lambda h: (h["evalue"], -h["bits"]))
    for level, passes in rules:
        h = next(filter(passes, hits), None)
        if h:
            return {"cath_label": truncate(cath_labels[h["target"]], level), "level": level, "method": "foldseek",
                    "evidence": {"cath_domain": h["target"], "evalue": h["evalue"], "tm": h["tm"],
                                 "coverage": h["coverage"], "prob": h["prob"], "fident": h["fident"]}}
    return None


class Foldclass:
    """Merizo-search's Foldclass: the nearest CATH S40 domain by structure embedding, confirmed with TM-align."""

    def __init__(self, device: str = "cpu"):
        import torch
        sys.path.insert(0, str(config.MERIZO_SEARCH))
        from programs.Foldclass import utils
        from programs.Foldclass.nndef_fold_egnn_embed import FoldClassNet

        self.torch, self.utils, self.device = torch, utils, torch.device(device)
        self.net = FoldClassNet(128).eval().to(self.device)
        weights = config.MERIZO_SEARCH / "programs/Foldclass/FINAL_foldclass_model.pt"
        self.net.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
        self.embeddings = torch.load(config.FOLDCLASS_DB.with_suffix(".pt"), map_location="cpu").to(self.device)
        with open(config.FOLDCLASS_DB.with_suffix(".index"), "rb") as fh:
            self.index = pickle.load(fh)                    # (file name, CA coordinates, sequence) per embedding

    def search(self, pdb: Path, cath_labels: dict, workdir: Path) -> dict:
        """A T-level label if the nearest CATH domain aligns with TM-score > 0.5, else an unlabelled result."""
        torch, utils = self.torch, self.utils
        query = utils.read_pdb(str(pdb), pdb_chain="A")
        with torch.no_grad():
            embedding = self.net(torch.from_numpy(query["coords"]).unsqueeze(0).to(self.device))
            similarity = torch.nn.functional.cosine_similarity(self.embeddings, embedding, dim=-1)
            cosine, nearest = torch.max(similarity, dim=0)
        name, coords, seq = self.index[int(nearest)]
        target_file = utils.write_pdb(str(workdir), coords, seq)
        query_file = utils.write_pdb(str(workdir), query["coords"], query["seq"])
        tm = utils.extract_tmalign_values(
            subprocess.run([str(config.TMALIGN), query_file, target_file], capture_output=True, text=True).stdout)
        os.remove(query_file)
        os.remove(target_file)
        domain = Path(name).stem
        passed = tm["qtm"] > FOLDCLASS_TM
        return {"cath_label": truncate(cath_labels[domain], "T") if passed else None, "level": "T" if passed else "-",
                "method": "foldclass",
                "evidence": {"cath_domain": domain, "cosine": float(cosine), "q_tm": tm["qtm"], "t_tm": tm["ttm"],
                             "rmsd": tm["rmsd"], "len_ali": tm["len_ali"]}}


# --- sequence: MMseqs2 against the representatives of TED's sequence clusters --------------------------------------
def _parse_rep_header(header: str) -> dict:
    """'REP TOP_LABEL LEVEL n=.. nlab=.. share=.. own=LABEL:METHOD', as written by scripts/build_ted_mmseqs_db.py"""
    rep, label, level, *pairs = header.split()
    kv = dict(p.split("=", 1) for p in pairs)
    rep_label, rep_method = kv["own"].split(":")
    return {"ted_rep": rep, "cluster_label": label, "cluster_level": level, "n_members": int(kv["n"]),
            "n_labelled": int(kv["nlab"]), "label_share": float(kv["share"]), "rep_label": rep_label,
            "rep_method": rep_method}


def mmseqs_search(seqs: dict[str, str], workdir: Path, threads: int = 8, gpu: bool = False,
                  server: bool = False) -> dict[str, list[dict]]:
    """Search sequences against the 120.7M representatives of TED's sequence clusters. Returns {name: hits}.

    gpu     use MMseqs2-GPU on the padded DB instead of the k-mer index on CPU
    server  the DB is already held in memory (page cache on CPU, `mmseqs gpuserver` on GPU), so map it, don't load it
    """
    db = Path(f"{config.TED_MMSEQS_DB}_pad" if gpu else config.TED_MMSEQS_DB)
    if not db.with_name(db.name + ".dbtype").exists():
        raise FileNotFoundError(f"TED MMseqs2 DB not built yet ({db}); run scripts/build_lookup_dbs.sbatch")
    mmseqs = config.MMSEQS_GPU if gpu else config.MMSEQS
    query, result, table = workdir / "query", workdir / "result", workdir / "result.tsv"
    (workdir / "query.fasta").write_text("".join(f">{name}\n{seq}\n" for name, seq in seqs.items()))
    common = ("--db-load-mode", 2 if server else 0, "--threads", threads, "-v", 1)
    how = ("--gpu", 1, "--gpu-server", int(server)) if gpu else ("-s", 7.5)
    _run(mmseqs, "createdb", workdir / "query.fasta", query, "-v", 1)
    _run(mmseqs, "search", query, db, result, workdir / "mm_tmp", *how, "--max-seqs", 300, "-e", "1e-3", "-a", *common)
    _run(mmseqs, "convertalis", query, db, result, table, "--format-output", MM_COLUMNS, *common)
    hits = {name: [] for name in seqs}
    for h in _read_hits(table, MM_COLUMNS, 3):
        h.update(_parse_rep_header(h.pop("theader")))
        h["lenratio"] = min(h["qlen"], h["tlen"]) / max(h["qlen"], h["tlen"])
        h["joins_ted_cluster"] = h["fident"] >= CLUSTER_IDENTITY and h["lenratio"] >= CLUSTER_LENGTH_RATIO
        hits[h["query"]].append(h)
    return hits


def server_url() -> str | None:
    """Address of a running MMseqs server (ted_recreate.server writes it to db/mmseqs/SERVER), if it answers."""
    try:
        url = json.loads(config.MMSEQS_SERVER_FILE.read_text())["url"]
        with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
            return url if r.status == 200 else None
    except (OSError, ValueError, KeyError):
        return None


def remote_search(url: str, seqs: dict[str, str]) -> dict[str, list[dict]]:
    """mmseqs_search, run by the server."""
    request = urllib.request.Request(f"{url}/search", data=json.dumps({"sequences": seqs}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as r:
        return json.load(r)["hits"]


def _in_gpu_job() -> bool:
    """A GPU we may use for a one-off search: only inside a Slurm job, because login nodes share theirs."""
    if not os.environ.get("SLURM_JOB_ID"):
        return False
    import torch
    return torch.cuda.is_available()


def sequence_transfer(hits: list[dict]) -> dict | None:
    """The label of the TED cluster this sequence would have joined, or None if it joins none.

    TED's clusters come from `mmseqs easy-linclust --min-seq-id 0.5 -c 0.9 --cov-mode 5`: a member aligns to its
    representative with >= 50% identity, and the shorter of the two is >= 90% as long as the longer. A query that
    meets this against a representative would have been a member, and TED gave members the labels of their labelled
    cluster-mates. If several representatives qualify, the best bit score wins (a sequence joins one cluster).
    The cluster may itself be unlabelled, in which case cath_label is None."""
    joins = [h for h in hits if h["joins_ted_cluster"]]
    if not joins:
        return None
    h = max(joins, key=lambda h: h["bits"])
    label = None if h["cluster_label"] == "-" else h["cluster_label"]
    return {"cath_label": label, "level": h["cluster_level"] if label else "-", "method": "mmseqs-transfer",
            "evidence": {"ted_rep": h["ted_rep"], "fident": h["fident"], "lenratio": h["lenratio"],
                         "qcov": h["qcov"], "tcov": h["tcov"], "evalue": h["evalue"], "bits": h["bits"],
                         "cluster_size": h["n_members"], "cluster_labelled": h["n_labelled"],
                         "cluster_label_share": h["label_share"], "rep_own_label": h["rep_label"],
                         "rep_label_method": h["rep_method"], "n_qualifying_clusters": len(joins)}}


# --- putting the three together ------------------------------------------------------------------------------------
@dataclass
class LabelResult:
    name: str
    sequence: str
    cath_label: str | None = None       # '3.40.50.300' (H level), '3.40.50' (T level) or None
    level: str = "-"                    # 'H', 'T' or '-'
    source: str = "none"                # what decided: ted-exact / foldseek / foldclass / mmseqs-transfer / none
    name_of_label: str | None = None
    ted_exact: list = field(default_factory=list)       # TED domains with this exact sequence
    structure: dict | None = None                       # what the structure tier found
    sequence_transfer: dict | None = None               # the TED cluster this sequence joins
    sequence_hits: list = field(default_factory=list)   # best 25 cluster representatives
    remote_hint: dict | None = None                     # nearest labelled homolog, when nothing gave a label
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Classifier:
    """Loads the reference data once, then labels any number of domains."""

    def __init__(self, tiers=TIERS, device: str = "cpu", threads: int = 8):
        self.tiers, self.threads = tuple(tiers), threads
        self.cath_labels = _read_map(config.CATH_LABELS, 1)
        self.cath_names = _read_map(config.CATH_NAMES, 2)
        self.md5_index = TedMd5Index() if "exact" in self.tiers else None
        self.foldclass = Foldclass(device) if "structure" in self.tiers else None

    def label(self, items: list[dict], workdir: str | Path | None = None) -> list[LabelResult]:
        """items: dicts with 'name', 'sequence' and, for the structure tier, 'pdb' (the domain's structure)."""
        results = {it["name"]: LabelResult(name=it["name"], sequence=it["sequence"]) for it in items}
        with work_directory(workdir, "ted_cath_") as work:
            if self.md5_index:
                for r in results.values():
                    r.ted_exact = self.md5_index.lookup(r.sequence)
            if self.foldclass:
                self._structure({it["name"]: it["pdb"] for it in items if it.get("pdb")}, results, work)
            if "sequence" in self.tiers:
                self._sequence(results, work)
        for r in results.values():
            self._decide(r)
        return list(results.values())

    def _structure(self, pdbs: dict, results: dict, work: Path):
        if not pdbs:
            for r in results.values():
                r.notes.append("no structure given: structure tier skipped")
            return
        queries = work / "domains"
        queries.mkdir(exist_ok=True)
        for name, pdb in pdbs.items():
            shutil.copy(pdb, queries / f"{name}.pdb")
        hits = foldseek_search(queries, work, self.threads)
        for name in pdbs:
            results[name].structure = (ted_foldseek_rule(hits.get(name, []), self.cath_labels)
                                       or self.foldclass.search(queries / f"{name}.pdb", self.cath_labels, work))

    def _sequence(self, results: dict, work: Path):
        seqs = {name: r.sequence for name, r in results.items()}
        url = server_url()
        if url:
            hits = remote_search(url, seqs)
        elif _in_gpu_job():         # no server: search once on this job's GPU (loads the 15 GB padded DB)
            hits = mmseqs_search(seqs, work, self.threads, gpu=True)
        else:
            for r in results.values():
                r.notes.append("sequence tier skipped: no MMseqs server is running (start one with "
                               "`sbatch scripts/mmseqs_server.sbatch`) and this is not a GPU job")
            return
        for name, hs in hits.items():
            results[name].sequence_hits = sorted(hs, key=lambda h: -h["bits"])[:25]

    def _decide(self, r: LabelResult):
        """The first source with a label decides: TED's own label, then structure, then TED's sequence cluster."""
        labelled_exact = [e for e in r.ted_exact if e["level"] != "-"]
        r.sequence_transfer = sequence_transfer(r.sequence_hits)        # reported whichever source decides
        if labelled_exact:
            decision, source = max(labelled_exact, key=lambda e: e["level"] == "H"), "ted-exact"
        elif r.structure and r.structure["cath_label"]:
            decision, source = r.structure, r.structure["method"]
        elif r.sequence_transfer and r.sequence_transfer["cath_label"]:
            decision, source = r.sequence_transfer, "mmseqs-transfer"
        else:
            decision = None
        if decision:
            r.cath_label, r.level, r.source = decision["cath_label"], decision["level"], source
            r.name_of_label = self.cath_names.get(r.cath_label)
        else:
            if r.sequence_transfer:
                r.notes.append("joins a TED sequence cluster that TED left unlabelled")
            homolog = next((h for h in r.sequence_hits if h["cluster_label"] != "-"), None)
            if homolog:
                r.remote_hint = {"cath_label": homolog["cluster_label"], "level": homolog["cluster_level"],
                                 "ted_rep": homolog["ted_rep"], "fident": homolog["fident"], "qcov": homolog["qcov"],
                                 "evalue": homolog["evalue"],
                                 "note": "homolog below TED's cluster criterion; not used as the label"}
        if r.ted_exact and not labelled_exact:
            r.notes.append("sequence is identical to TED domain(s) that TED left unlabelled")


def write_domain_pdb(chain_pdb: str | Path, segments, out: Path) -> Path:
    """Cut a domain out of a chain model. `segments` are residue ranges as in a chopping string."""
    keep = {r for a, b in segments for r in range(a, b + 1)}
    with open(chain_pdb) as src, open(out, "w") as dst:
        dst.writelines(line for line in src if line.startswith("ATOM") and int(line[22:26]) in keep)
        dst.write("END\n")
    return out


def cath_label(sequence: str | None = None, *, pdb: str | Path | None = None, name: str = "query",
               tiers=TIERS, **kw) -> LabelResult:
    """CATH label for one domain. Give its sequence, or a PDB of the domain, which also enables the structure tier."""
    if sequence is None and pdb is None:
        raise ValueError("give a sequence and/or pdb=")
    sequence = "".join(sequence.split()).upper() if sequence else read_pdb_ca(pdb)[0]
    return Classifier(tiers, **kw).label([{"name": name, "sequence": sequence, "pdb": pdb}])[0]


def cath_label_chopped(chop_results, tiers=TIERS, workdir: str | Path | None = None, **kw) -> list[LabelResult]:
    """Label every domain of ted_chop() results. Domain structures are cut from the chain models, where those exist."""
    with work_directory(workdir, "ted_cath_") as work:
        items = []
        for r in chop_results:
            for d in r.domains:
                item = {"name": d.ted_id, "sequence": d.sequence}
                if r.structure and Path(r.structure).exists():
                    item["pdb"] = write_domain_pdb(r.structure, parse_chopping(d.chopping)[0], work / f"{d.ted_id}.pdb")
                items.append(item)
        return Classifier(tiers, **kw).label(items, workdir=work / "work")
