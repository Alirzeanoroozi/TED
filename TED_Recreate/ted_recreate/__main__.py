"""Command line for the TED re-creation.

  python -m ted_recreate chop  --fasta proteins.fa       sequences, folded with ESMFold      -> TED domains
  python -m ted_recreate chop  --uniprot P69905 P00533   AlphaFold DB models (TED's input)   -> TED domains
  python -m ted_recreate chop  --pdb model.pdb           your own model                      -> TED domains
  python -m ted_recreate label --fasta proteins.fa       chop, then a CATH label for every domain
  python -m ted_recreate cath  --fasta domains.fa        domain sequences                    -> CATH labels
  python -m ted_recreate cath  --pdb domain.pdb          domain structures                   -> CATH labels

chop and label need a GPU node for anything but a few small proteins: sbatch scripts/gpu.sbatch -m ted_recreate ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

from .chop import ted_chop_batch, work_directory
from .classify import TIERS, Classifier, cath_label_chopped
from .structure import read_pdb_ca


def read_fasta(path) -> dict[str, str]:
    records, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            records[name] = ""
        elif name:
            records[name] += line
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ted_recreate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["chop", "cath", "label"])
    ap.add_argument("--fasta")
    ap.add_argument("--seq", help="a single sequence")
    ap.add_argument("--uniprot", nargs="+", default=[])
    ap.add_argument("--pdb", nargs="+", default=[])
    ap.add_argument("--tiers", default=",".join(TIERS), help="label sources to use (default: %(default)s)")
    ap.add_argument("--workdir", help="keep the models and intermediate files here")
    ap.add_argument("--out", help="write the full results as JSON")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)

    items = [{"name": name, "sequence": seq} for name, seq in (read_fasta(a.fasta) if a.fasta else {}).items()]
    if a.seq:
        items.append({"name": "query", "sequence": a.seq})
    items += [{"uniprot": acc} for acc in a.uniprot]
    items += [{"name": Path(p).stem, "pdb": p} for p in a.pdb]
    if not items:
        ap.error("give --fasta, --seq, --uniprot or --pdb")
    if a.command == "cath" and a.uniprot:
        ap.error("cath takes domain sequences (--fasta, --seq) or domain structures (--pdb)")
    tiers = [t for t in a.tiers.split(",") if t]

    chopped, labels = [], []
    with work_directory(a.workdir) as work:     # the structure tier cuts domains out of the models: keep them for now
        if a.command == "cath":
            for item in items:
                if "pdb" in item:
                    item["sequence"] = read_pdb_ca(item["pdb"])[0]
                item["sequence"] = "".join(item["sequence"].split()).upper()
            labels = Classifier(tiers, threads=a.threads).label(items, workdir=work / "cath")
        else:
            chopped = ted_chop_batch(items, workdir=work)
            for r in chopped:
                domains = ", ".join(f"{d.ted_id.rsplit('_', 1)[-1]}:{d.chopping}({d.consensus_level[0]})"
                                    for d in r.domains)
                print(f"{r.name}\t{r.nres} aa\t{r.structure_source}\t{domains or 'no TED domains'}")
                for w in r.warnings:
                    print(f"  WARNING {r.name}: {w}", file=sys.stderr)
            if a.command == "label":
                labels = cath_label_chopped(chopped, tiers, threads=a.threads, workdir=work / "cath")
    for l in labels:
        print(f"{l.name}\t{l.cath_label or '-'}\t{l.level}\t{l.source}\t{l.name_of_label or ''}")
    for r in chopped:
        if not Path(r.structure).exists():
            r.structure = ""                    # the model was temporary; pass --workdir to keep it
    out = {}
    if a.command != "cath":
        out["chop"] = [r.to_dict() for r in chopped]
    if a.command != "chop":
        out["cath"] = [l.to_dict() for l in labels]
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))
        print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
    # After ESMFold has used the GPU the interpreter can hang while shutting down (seen in Slurm jobs). Everything
    # is written by now, so leave without waiting for library threads.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
