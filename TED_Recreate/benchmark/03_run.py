"""Run our functions on the benchmark proteins. GPU job: sbatch scripts/gpu.sbatch benchmark/03_run.py

For every protein in proteins.json:
  afdb mode   ted_chop(pdb = AlphaFold model TED used)        -> chop_afdb.json
  seq  mode   ted_chop(sequence = ...)  (ESMFold, then chop)   -> chop_esm.json
and CATH labels (all three tiers, each recorded separately) for three sets of domains:
  labels_ted.json    TED's published domains, cut from the AlphaFold model
  labels_afdb.json   our domains from the AlphaFold model
  labels_esm.json    our domains from the ESMFold model
Also the TM-score between each ESMFold model and its AlphaFold model (tm_esm_vs_afdb.json), each TED domain's own
sequence cluster (own_cluster.tsv, for the "as if new to TED" scores) and stage timings.
Each stage is skipped if its output exists, so the job can be resubmitted.
"""
import json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ted_recreate import config
from ted_recreate.chop import ChopResult, Domain, parse_chopping, ted_chop_batch
from ted_recreate.classify import Classifier, write_domain_pdb

here = Path(__file__).parent
prot = json.load(open(here / "proteins.json"))
timings = json.load(open(here / "timings.json")) if (here / "timings.json").exists() else {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def chop_safe(items, workdir, chunk=100):
    """ted_chop_batch in chunks; a chunk that fails is retried protein by protein so one bad model costs one protein."""
    results, failed = [], {}
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        try:
            results += ted_chop_batch(part, workdir=workdir / f"chunk{i:04d}")
        except Exception as e:
            log(f"chunk {i} failed ({repr(e)[:200]}); retrying one by one")
            for k, it in enumerate(part):
                try:
                    results += ted_chop_batch([it], workdir=workdir / f"chunk{i:04d}_single{k:03d}")
                except Exception as e2:
                    failed[it["name"]] = repr(e2)[:300]
                    log(f"  {it['name']} failed: {repr(e2)[:200]}")
        log(f"  {len(results)}/{len(items)} done, {len(failed)} failed")
    return results, failed


def load_results(path):
    out = []
    for d in json.load(open(path))["results"]:
        d = dict(d)
        d.pop("residue_labels", None)
        doms = [Domain(**x) for x in d.pop("domains")]
        r = ChopResult(**d)
        r.domains = doms
        out.append(r)
    return out


def stage_chop(tag, items, workdir):
    out = here / f"chop_{tag}.json"
    if out.exists():
        return load_results(out)
    t = time.time()
    res, failed = chop_safe(items, workdir)
    timings[f"chop_{tag}_seconds"] = time.time() - t
    timings[f"chop_{tag}_n"] = len(res)
    json.dump({"results": [r.to_dict() for r in res], "failed": failed}, open(out, "w"))
    json.dump(timings, open(here / "timings.json", "w"), indent=1)
    return res


log(f"{len(prot)} proteins")
afdb = stage_chop("afdb", [{"pdb": v["pdb"], "name": c} for c, v in prot.items()], here / "work_afdb")
esm = stage_chop("esm", [{"sequence": v["sequence"], "name": c + "_esm"} for c, v in prot.items()], here / "work_esm")

# --- TM-score ESMFold vs AlphaFold model (how different is the structure we chop?) ---
tm_out = here / "tm_esm_vs_afdb.json"
if not tm_out.exists():
    def tm(r):
        chain = r.name[:-4]
        p = subprocess.run([str(config.TMALIGN), r.structure, prot[chain]["pdb"]], capture_output=True, text=True)
        s = re.findall(r"TM-score= ([0-9.]+)", p.stdout)
        rmsd = re.search(r"RMSD=\s*([0-9.]+)", p.stdout)
        return chain, {"tm": float(s[0]) if s else None, "rmsd": float(rmsd.group(1)) if rmsd else None}
    with ThreadPoolExecutor(8) as ex:
        json.dump(dict(ex.map(tm, esm)), open(tm_out, "w"), indent=1)
    log("TM-scores done")

# --- each TED domain's own sequence cluster: rows (representative, member, label, method) of TED's cluster table ---
own_out = here / "own_cluster.tsv"
if not own_out.exists():
    ids = here / "ted_domain_ids.txt"
    ids.write_text("".join(d["ted_id"] + "\n" for v in prot.values() for d in v["ted"]))
    clusters = config.TED_MMSEQS_DB.parent / "clusters.sorted.tsv"       # written by scripts/build_ted_mmseqs_db.py
    subprocess.run(f"awk -F'\\t' 'NR==FNR {{want[$1]=1; next}} ($2 in want)' {ids} {clusters} > {own_out}.part "
                   f"&& mv {own_out}.part {own_out}", shell=True, check=True, env={**os.environ, "LC_ALL": "C"})
    log("own clusters done")

# --- CATH labels ---
cls = None


def stage_label(tag, items):
    global cls
    out = here / f"labels_{tag}.json"
    if out.exists() or not items:
        return
    if cls is None:
        cls = Classifier(threads=8)
    t = time.time()
    res = []
    for i in range(0, len(items), 400):
        res += cls.label(items[i:i + 400])
        log(f"  labels {tag}: {len(res)}/{len(items)}")
    timings[f"label_{tag}_seconds"] = time.time() - t
    timings[f"label_{tag}_n"] = len(items)
    json.dump([r.to_dict() for r in res], open(out, "w"))
    json.dump(timings, open(here / "timings.json", "w"), indent=1)


dom_dir = here / "domains"
for sub in ("ted", "afdb", "esm"):
    (dom_dir / sub).mkdir(parents=True, exist_ok=True)

items = []
for chain, v in prot.items():
    for d in v["ted"]:
        segs = parse_chopping(d["chopping"])[0]
        pdb = write_domain_pdb(v["pdb"], segs, dom_dir / "ted" / f"{d['ted_id']}.pdb")
        seq = "".join(v["sequence"][a - 1:b] for a, b in segs)
        items.append({"name": d["ted_id"], "sequence": seq, "pdb": str(pdb)})
stage_label("ted", items)

for tag, results in (("afdb", afdb), ("esm", esm)):
    items = []
    for r in results:
        for d in r.domains:
            pdb = write_domain_pdb(r.structure, parse_chopping(d.chopping)[0], dom_dir / tag / f"{d.ted_id}.pdb")
            items.append({"name": d.ted_id, "sequence": d.sequence, "pdb": str(pdb)})
    stage_label(tag, items)
log("all stages done")
# after CUDA work the interpreter can hang in teardown and keep the GPU; everything is written, so leave now
sys.stdout.flush()
os._exit(0)
