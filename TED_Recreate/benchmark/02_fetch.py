"""Fetch the AlphaFold models of the sampled proteins and keep those TED's domains can be verified on.

A protein is usable when the AFDB model's sequence reproduces every published md5_domain (i.e. it is still
the sequence TED chopped) and it is at most MAXLEN residues (ESMFold on one A100).
Output: afdb/<chain>.pdb, proteins.json (the benchmark set, in a fixed random order), fetch_log.json
"""
import json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ted_recreate.chop import cut, parse_chopping
from ted_recreate.structure import fetch_afdb_model, md5, read_pdb_ca

here = Path(__file__).parent
N, MAXLEN = int(sys.argv[1]), int(sys.argv[2])
cand = json.load(open(here / "sample_candidates.json"))
chains = sorted(cand)
random.Random(42).shuffle(chains)

def get(chain):
    acc = chain.split("-")[1]
    try:
        pdb, rec = fetch_afdb_model(acc, here / "afdb")
        seq, plddt = read_pdb_ca(pdb)
        ok = all(md5(cut(seq, parse_chopping(d["chopping"])[0])) == d["md5_domain"] for d in cand[chain])
        return chain, {"pdb": str(pdb), "nres": len(seq), "sequence": seq, "seq_ok": ok,
                       "afdb_mean_plddt": sum(plddt) / len(plddt), "model_created": rec.get("modelCreatedDate"),
                       "afdb_version": rec.get("latestVersion")}
    except Exception as e:
        return chain, {"error": repr(e)[:120]}

with ThreadPoolExecutor(8) as ex:
    log = dict(ex.map(get, chains))
json.dump(log, open(here / "fetch_log.json", "w"), indent=1)
n_err = sum("error" in v for v in log.values())
n_changed = sum(("seq_ok" in v and not v["seq_ok"]) for v in log.values())
ok = [c for c in chains if log[c].get("seq_ok")]
too_long = [c for c in ok if log[c]["nres"] > MAXLEN]
usable = [c for c in ok if log[c]["nres"] <= MAXLEN][:N]
print(f"candidates {len(chains)}: no AFDB model {n_err}, sequence changed since TED {n_changed}, verified {len(ok)}, "
      f"of which longer than {MAXLEN} aa: {len(too_long)} ({len(too_long)/max(1,len(ok)):.1%}); benchmark set {len(usable)}")
json.dump({c: {**log[c], "ted": cand[c]} for c in usable}, open(here / "proteins.json", "w"), indent=1)
