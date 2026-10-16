"""Build the CATH 4.3 reference files used by ted_recreate.classify.

TED searched its domains against 31,574 CATH 4.3 'SSG5' representatives. That list is not public, so we
use the closest public set: the CATH 4.3 S40 non-redundant domains (31,885), which gives Foldseek E-values
on essentially the same scale. Outputs (db/cath/):
  cath43_labels.tsv          domain_id -> C.A.T.H label for all CATH 4.3 domains (+ S40 flag)
  cath43_names.tsv           CATH node -> name
  foldclass_s40.pt/.index    Merizo-search Foldclass embeddings + CA coords restricted to S40 domains
"""
import pickle
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
cath = ROOT / "data" / "cath"
out = ROOT / "db" / "cath"
out.mkdir(parents=True, exist_ok=True)

s40 = {l.strip() for l in open(cath / "cath-dataset-nonredundant-S40-v4_3_0.list") if l.strip()}
labels = {}
with open(cath / "cath-domain-list-v4_3_0.txt") as fh, open(out / "cath43_labels.tsv", "w") as fo:
    fo.write("domain\tcath\tin_s40\tlength\tresolution\n")
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split()
        lab = ".".join(f[1:5])
        labels[f[0]] = lab
        fo.write(f"{f[0]}\t{lab}\t{int(f[0] in s40)}\t{f[10]}\t{f[11]}\n")
print("CATH 4.3 domains:", len(labels), "S40:", len(s40), "S40 with label:", sum(d in labels for d in s40))

with open(cath / "cath-names-v4_3_0.txt") as fh, open(out / "cath43_names.tsv", "w") as fo:
    fo.write("node\trep_domain\tname\n")
    for line in fh:
        if line.startswith("#"):
            continue
        node, rep, name = line.rstrip("\n").split(None, 2)
        fo.write(f"{node}\t{rep}\t{name.lstrip(':')}\n")

ms = ROOT / "data" / "merizo_search_cath" / "cath-4.3-foldclassdb"
emb = torch.load(str(ms) + ".pt", map_location="cpu")
index = pickle.load(open(str(ms) + ".index", "rb"))
keep = [i for i, (name, _, _) in enumerate(index) if Path(name).stem in s40]
print("Foldclass DB entries:", len(index), "kept (S40):", len(keep))
torch.save(emb[keep].clone(), out / "foldclass_s40.pt")
pickle.dump([index[i] for i in keep], open(out / "foldclass_s40.index", "wb"))
