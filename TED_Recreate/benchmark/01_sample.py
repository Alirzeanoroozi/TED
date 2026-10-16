"""Draw a random sample of TED proteins with their published domains.

One pass over ted_365m.domain_summary: a protein is taken if crc32(chain id) % MOD == 0, which spreads the sample
evenly over the whole (accession-sorted) database. TED-redundant chains are dropped afterwards (they were made
with a different consensus function), leaving TED-100 proteins.
Output: sample_candidates.json  {chain: [ {ted_id, md5_domain, consensus_level, chopping, nres, plddt,
                                           cath_label, cath_level, cath_method}, ... ]}
"""
import gzip, json, subprocess, sys, zlib
from pathlib import Path
R = Path(__file__).resolve().parents[1]
MOD = int(sys.argv[1]) if len(sys.argv) > 1 else 150000
src = R / "data/ted/ted_365m.domain_summary.cath.globularity.taxid.tsv.gz"
p = subprocess.Popen(f"zcat {src} | cut -f1-7,13-16,19-21", shell=True, stdout=subprocess.PIPE, bufsize=1 << 24)
sample, cur, keep, n = {}, None, False, 0
for line in p.stdout:
    n += 1
    tid = line[:line.index(b"\t")]
    chain = tid[:tid.rindex(b"_TED")]
    if chain != cur:
        cur = chain
        keep = zlib.crc32(chain) % MOD == 0
    if keep:
        f = line.decode().rstrip("\n").split("\t")
        sample.setdefault(chain.decode(), []).append({
            "ted_id": f[0], "md5_domain": f[1], "consensus_level": f[2], "chopping": f[3], "nres": int(f[4]),
            "num_segments": int(f[5]), "plddt": float(f[6]), "proteome": f[7], "cath_label": f[8], "cath_level": f[9],
            "cath_method": f[10], "species": f[12], "lineage": f[13]})
    if n % 50_000_000 == 0:
        print(f"{n:,} rows, {len(sample)} proteins sampled", flush=True)
print(f"{n:,} rows, {len(sample)} proteins sampled", flush=True)
ids = {d["ted_id"] for ds in sample.values() for d in ds}
red = set()
with gzip.open(R / "data/ted/ted_redundant_40m_domain_id.list.gz", "rt") as fh:
    for line in fh:
        if line.strip() in ids:
            red.add(line.strip().rsplit("_TED", 1)[0])
ted100 = {c: ds for c, ds in sample.items() if c not in red}
print(f"TED-redundant proteins dropped: {len(red)}; TED-100 proteins kept: {len(ted100)} with {sum(map(len, ted100.values()))} domains")
json.dump(ted100, open(Path(__file__).parent / "sample_candidates.json", "w"), indent=1)
