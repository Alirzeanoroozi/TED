"""Build db/ted_md5/: an md5_domain -> (TED id, CATH label, level, method) table for all 365M TED domains.

Input: data/ted/ted_365m.domain_summary.cath.globularity.taxid.tsv.gz (Zenodo 13908086), columns used:
  1 ted_id, 2 md5_domain, 14 cath_label, 15 cath_assignment_level (H/T/-), 16 cath_assignment_method.
Output (numpy .npy, read with mmap by ted_recreate.classify.TedMd5Index):
  md5_hi / md5_lo   uint64 halves of the md5, sorted
  label             int32 index into labels.json["labels"]
  level             uint8 0 '-', 1 'T', 2 'H'
  method            uint8 index into labels.json["methods"]
  ted_row           int64 row in ted_ids.txt (original file order)
  ted_ids.txt + ted_ids.offsets.npy   the TED ids, one per line, with byte offsets
Needs ~40 GB RAM; run through slurm (scripts/build_lookup_dbs.sbatch).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/ted/ted_365m.domain_summary.cath.globularity.taxid.tsv.gz"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "db/ted_md5"
out.mkdir(parents=True, exist_ok=True)

LEVEL = {"-": 0, "T": 1, "H": 2}
labels, methods = {"-": 0}, {"-": 0}
his, los, labs, levs, meths, offs = [], [], [], [], [], []
offset = 0
t0 = time.time()
n = 0
with open(out / "ted_ids.txt", "wb") as fid:
    reader = pd.read_csv(src, sep="\t", header=None, usecols=[0, 1, 13, 14, 15], dtype=str,
                         chunksize=5_000_000, na_filter=False, engine="c")
    for chunk in reader:
        ids = chunk[0].to_numpy()
        blob = ("\n".join(ids) + "\n").encode()
        fid.write(blob)
        lens = np.fromiter((len(s) + 1 for s in ids), dtype=np.int64, count=len(ids))
        offs.append(offset + np.concatenate([[0], np.cumsum(lens)[:-1]]))
        offset += int(lens.sum())
        raw = np.frombuffer(bytes.fromhex("".join(chunk[1].to_numpy())), dtype=">u8").reshape(-1, 2)
        his.append(raw[:, 0].astype(np.uint64))
        los.append(raw[:, 1].astype(np.uint64))
        labs.append(np.array([labels.setdefault(x, len(labels)) for x in chunk[13].to_numpy()], dtype=np.int32))
        levs.append(np.array([LEVEL.get(x, 0) for x in chunk[14].to_numpy()], dtype=np.uint8))
        meths.append(np.array([methods.setdefault(x, len(methods)) for x in chunk[15].to_numpy()], dtype=np.uint8))
        n += len(ids)
        print(f"{n:,} rows  {time.time()-t0:.0f}s", flush=True)

offs.append(np.array([offset], dtype=np.int64))
np.save(out / "ted_ids.offsets.npy", np.concatenate(offs))
hi, lo = np.concatenate(his), np.concatenate(los)
del his, los
print("sorting", flush=True)
order = np.lexsort((lo, hi))
np.save(out / "md5_hi.npy", hi[order]); del hi
np.save(out / "md5_lo.npy", lo[order]); del lo
np.save(out / "label.npy", np.concatenate(labs)[order])
np.save(out / "level.npy", np.concatenate(levs)[order])
np.save(out / "method.npy", np.concatenate(meths)[order])
np.save(out / "ted_row.npy", order.astype(np.int64))
inv = lambda d: [k for k, _ in sorted(d.items(), key=lambda kv: kv[1])]
json.dump({"labels": inv(labels), "methods": inv(methods), "n_rows": n, "source": str(src)}, open(out / "labels.json", "w"))
print(f"done: {n:,} domains, {len(labels):,} distinct labels, {time.time()-t0:.0f}s", flush=True)
