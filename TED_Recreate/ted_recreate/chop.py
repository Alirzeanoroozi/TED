"""TED-style domain chopping: structure -> Merizo + Chainsaw + UniDoc -> consensus domains.

The segmentation is TED's own released code, run unmodified (vendor/ted-tools/ted_consensus_1.0/run_segmentation.sh):
  1. Merizo on the full model
  2. UniDoc on the residues Merizo put in domains
  3. Chainsaw on the full model (+1 residue offset, Chainsaw is 0-based)
  each followed by a filter that drops segments < 5 residues and domains < 25 residues
  4. consensus: domains from different parsers are linked when their IoU >= 0.7; a connected group of 3 is "high",
     2 "medium", 1 "low"; the range kept is the residues shared by the whole group; then the same filter again.
Only high and medium domains are TED domains.

Numbering: TED's consensus code adds 1 to every residue index before intersecting (domain_consensus.py:
domstr_to_assignment(..., offset=1)), so consensus ranges sit one residue after the parsers' ranges. The published
TED database has the same shift, so `chopping` is in TED's convention and `chopping_parser_frame` undoes the shift.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config
from .structure import ESMFold, clean_sequence, fetch_afdb_model, md5, normalise_pdb, read_pdb_ca

PARSERS = ("merizo", "chainsaw", "unidoc")


@dataclass
class Domain:
    ted_id: str
    consensus_level: str            # high (all three parsers agree) or medium (two agree)
    chopping: str                   # TED's convention, the frame of the TED database
    chopping_parser_frame: str      # one residue earlier: what the parsers actually agreed on
    nres: int
    num_segments: int
    sequence: str
    md5_domain: str
    plddt: float | None


@dataclass
class ChopResult:
    name: str
    sequence: str
    md5: str
    nres: int
    structure: str                  # path of the model that was chopped ("" if it was temporary)
    structure_source: str
    methods: dict = field(default_factory=dict)         # merizo / chainsaw / unidoc -> chopping string
    consensus: dict = field(default_factory=dict)       # high / medium / low -> chopping string
    domains: list = field(default_factory=list)         # the TED domains (high + medium), TED01.. from the N-terminus
    low_consensus: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def residue_labels(self) -> list[int]:
        """Domain number of every residue (1 = TED01, ...), 0 where the residue is in no TED domain."""
        labels = [0] * self.nres
        for i, d in enumerate(self.domains, 1):
            for a, b in parse_chopping(d.chopping)[0]:
                labels[a - 1:b] = [i] * len(labels[a - 1:b])
        return labels

    def to_dict(self) -> dict:
        return {**asdict(self), "residue_labels": self.residue_labels()}


def parse_chopping(chopping: str) -> list[list[tuple[int, int]]]:
    """'11-41_290-389,54-288' -> [[(11, 41), (290, 389)], [(54, 288)]]; TED's "no domain" markers -> []"""
    if chopping in ("na", "NULL", "NO_SS", "0", "", None):
        return []
    return [[tuple(map(int, seg.split("-"))) for seg in dom.split("_")] for dom in chopping.split(",")]


def format_chopping(segments) -> str:
    return "_".join(f"{a}-{b}" for a, b in segments)


def cut(seq: str, segments) -> str:
    """The residues of `seq` covered by the segments (1-based, inclusive)."""
    return "".join(seq[a - 1:b] for a, b in segments)


@contextmanager
def work_directory(workdir=None, prefix="ted_"):
    """`workdir` (created if needed, and kept) or, without one, a temporary directory that is removed afterwards."""
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
        yield Path(workdir)
    else:
        with tempfile.TemporaryDirectory(prefix=prefix, ignore_cleanup_errors=True) as tmp:
            yield Path(tmp)


def _read_table(path: Path) -> dict[str, list[str]]:
    """Rows of one of TED's output tables, keyed by target name."""
    if not path.exists():
        return {}
    rows = (line.split("\t") for line in path.read_text().splitlines())
    return {f[0]: f for f in rows if f[0] and not f[0].startswith("Done")}


def run_ted_segmentation(pdbs: dict[str, Path], workdir: Path) -> dict[str, dict]:
    """Run TED's run_segmentation.sh on {name: model}. Returns the parsers' choppings and the consensus per name."""
    inp, out = workdir / "input", workdir / "out"
    inp.mkdir(parents=True, exist_ok=True)
    for name, pdb in pdbs.items():
        normalise_pdb(pdb, inp / f"{name}.pdb")
    venv = workdir / "ted_consensus"                # the script activates ./ted_consensus
    if not venv.exists():
        venv.symlink_to(config.VENV)
    env = {**os.environ, "PATH": f"{config.VENV / 'bin'}:{os.environ.get('PATH', '')}"}
    proc = subprocess.run(["bash", str(config.TED_CONSENSUS / "run_segmentation.sh"), "-i", "input", "-o", "out"],
                          cwd=workdir, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        logs = "\n".join(f"--- {p.name}\n{p.read_text()[-2000:]}" for p in sorted(out.glob("*.log")))
        raise RuntimeError(f"run_segmentation.sh failed ({proc.returncode})\n{proc.stdout}\n{proc.stderr}\n{logs}")
    parsers = {m: _read_table(out / f"chopping_{m}.txt") for m in PARSERS}
    consensus = _read_table(out / "consensus.tsv")
    results = {}
    for name in pdbs:
        row = consensus[name]
        results[name] = {"md5": row[1],
                         "methods": {m: rows[name][4] if name in rows else "NULL" for m, rows in parsers.items()},
                         "consensus": {"high": row[6], "medium": row[7], "low": row[8]}}
    return results


def _valid_name(name: str) -> str:
    """Names become file names and table keys in TED's scripts."""
    if not name or any(c in name for c in "/ \t,"):
        raise ValueError(f"bad name {name!r}: no slashes, spaces, tabs or commas")
    return name


def _models(items, outdir: Path, device) -> dict[str, tuple]:
    """A model for every item: {name: (sequence, model path, where the model came from)}."""
    outdir.mkdir(parents=True, exist_ok=True)
    models, esm = {}, None
    for item in items:
        if isinstance(item, str):
            item = {"sequence": item}
        if "pdb" in item or "uniprot" in item:
            if "pdb" in item:
                pdb, source = Path(item["pdb"]), "user_pdb"
            else:
                pdb, rec = fetch_afdb_model(item["uniprot"], outdir)
                source = f"AFDB v{rec.get('latestVersion')} ({rec['entryId']})"
            seq, name = read_pdb_ca(pdb)[0], _valid_name(item.get("name") or pdb.stem)
        else:
            seq = clean_sequence(item["sequence"])
            name = _valid_name(item.get("name") or f"query_{md5(seq)[:10]}")
            esm = esm or ESMFold(device)
            pdb = outdir / f"{name}.pdb"
            info = esm.fold(seq, pdb)
            source = f"ESMFold (mean pLDDT {info['mean_plddt']:.1f}, pTM {info['ptm']:.2f})"
        models[name] = (seq, pdb, source)
    if esm:
        esm.release()
    return models


def _assemble(name, seq, pdb, source, seg) -> ChopResult:
    """Turn TED's output for one protein into a ChopResult."""
    if seg["md5"] != md5(seq):
        raise RuntimeError(f"{name}: sequence in structure does not match input sequence")
    plddt = read_pdb_ca(pdb)[1]
    res = ChopResult(name=name, sequence=seq, md5=md5(seq), nres=len(seq), structure=str(pdb), structure_source=source,
                     methods=seg["methods"], consensus=seg["consensus"])
    accepted = sorted((segs[0][0], level, segs) for level in ("high", "medium")
                      for segs in parse_chopping(seg["consensus"][level]))
    for i, (_, level, segs) in enumerate(accepted, 1):
        dom_seq = cut(seq, segs)
        dom_plddt = [x for a, b in segs for x in plddt[a - 1:b]]
        res.domains.append(Domain(
            ted_id=f"{name}_TED{i:02d}", consensus_level=level, chopping=format_chopping(segs),
            chopping_parser_frame=format_chopping([(a - 1, b - 1) for a, b in segs]),
            nres=sum(b - a + 1 for a, b in segs), num_segments=len(segs), sequence=dom_seq, md5_domain=md5(dom_seq),
            plddt=round(sum(dom_plddt) / len(dom_plddt), 4) if dom_plddt else None))
    res.low_consensus = [format_chopping(segs) for segs in parse_chopping(seg["consensus"]["low"])]
    res.warnings = [f"{m} failed on this structure (NO_SS); consensus used the other parsers only"
                    for m, chopping in seg["methods"].items() if chopping == "NO_SS"]
    return res


def ted_chop_batch(items, workdir: str | Path | None = None, device: str | None = None) -> list[ChopResult]:
    """Chop many proteins in one go (the models are loaded once).

    items: a list in which each entry is one of
        {"sequence": "MKT...", "name": "optional_id"}   folded with ESMFold (a plain string works too)
        {"uniprot": "P69905"}                            the AlphaFold DB model, which is what TED chopped
        {"pdb": "path/to/model.pdb", "name": "..."}      your own model (first chain)
    workdir: keep the models and the parsers' raw output here; without it they are deleted.
    """
    with work_directory(workdir, "ted_chop_") as work:
        models = _models(items, work / "structures", device)
        seg = run_ted_segmentation({name: pdb for name, (_, pdb, _) in models.items()}, work / "segmentation")
        results = [_assemble(name, seq, pdb, source, seg[name]) for name, (seq, pdb, source) in models.items()]
        if workdir is None:
            for r in results:
                if work in Path(r.structure).parents:
                    r.structure = ""
        return results


def ted_chop(sequence: str | None = None, *, uniprot: str | None = None, pdb: str | None = None,
             name: str | None = None, **kw) -> ChopResult:
    """Chop one protein into TED consensus domains. Give exactly one of sequence, uniprot= or pdb=."""
    given = {k: v for k, v in (("sequence", sequence), ("uniprot", uniprot), ("pdb", pdb)) if v is not None}
    if len(given) != 1:
        raise ValueError("give exactly one of sequence, uniprot= or pdb=")
    return ted_chop_batch([{**given, "name": name}], **kw)[0]
