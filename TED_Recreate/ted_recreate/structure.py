"""A 3D model for a protein, so it can be chopped the TED way.

TED chopped AlphaFold DB v4 models. Given a UniProt accession we fetch the AFDB model (AFDB now serves v6 files; for
models made in the v4 era the coordinates are identical). A bare sequence has no AFDB model, so it is folded with
ESMFold. That is the one step that cannot match TED: ESMFold is not AlphaFold2, so the model, and with it the domain
boundaries, can differ, mostly in low-confidence regions.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from . import config

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def md5(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()


def clean_sequence(seq: str) -> str:
    seq = "".join(seq.split()).upper()
    bad = set(seq) - set(THREE_TO_ONE.values())
    if bad:
        raise ValueError(f"non-standard residues {sorted(bad)} in sequence; TED covers the 20 standard amino acids")
    return seq


def read_pdb_ca(pdb: str | Path) -> tuple[str, list[float]]:
    """Sequence and per-residue B-factor (pLDDT in predicted models) from the CA atoms of the first chain."""
    seq, bfactor, chain, seen = [], [], None, set()
    with open(pdb) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            chain = chain or line[21]
            if line[21] != chain or line[22:27] in seen:
                continue
            seen.add(line[22:27])
            seq.append(THREE_TO_ONE.get(line[17:20], "X"))
            bfactor.append(float(line[60:66]))
    return "".join(seq), bfactor


def normalise_pdb(src: str | Path, dst: str | Path) -> Path:
    """Write the first chain of the first model the way TED's tools expect it: chain A, residues numbered from 1,
    ATOM records only, then TER and END, every line padded to 80 columns (the layout of AFDB files).

    The padding matters: the UniDoc build TED used crashes on short lines such as ESMFold's 'PARENT N/A' header
    or a bare 'END'."""
    atoms, chain, residue, n = [], None, None, 0
    with open(src) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            chain = chain or line[21]
            if line[21] != chain:
                continue
            if line[22:27] != residue:
                residue, n = line[22:27], n + 1
            atoms.append(f"{line[:21]}A{n:4d} {line[27:].rstrip()}")
    last = atoms[-1]
    ter = f"TER   {int(last[6:11]) + 1:5d}      {last[17:20]} A{last[22:26]}"
    dst = Path(dst)
    dst.write_text("\n".join(line.ljust(80) for line in atoms + [ter, "END"]) + "\n")
    return dst


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ted-recreate"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch_afdb_model(uniprot: str, outdir: str | Path) -> tuple[Path, dict]:
    """Download the AlphaFold DB model of a UniProt accession. Returns (path, AFDB's record for the entry)."""
    records = json.loads(_download(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"))
    rec = next((r for r in records if r["entryId"].endswith("-F1")), records[0])
    path = Path(outdir) / f"{rec['entryId']}-model_v4.pdb"      # TED's file naming, so names line up with TED ids
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_download(rec["pdbUrl"]))
    return path, rec


class ESMFold:
    """The Hugging Face port of ESMFold. Loads once, folds many sequences."""

    def __init__(self, device: str | None = None):
        os.environ.setdefault("HF_HOME", str(config.HF_HOME))
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.ESMFOLD_MODEL)
        self.model = EsmForProteinFolding.from_pretrained(config.ESMFOLD_MODEL, low_cpu_mem_usage=True).eval()
        if self.device.startswith("cuda"):
            self.model.esm = self.model.esm.half()      # ESM-2 trunk in fp16, folding head in fp32
        self.model = self.model.to(self.device)
        self.model.trunk.set_chunk_size(64)             # bounds memory on long sequences

    def fold(self, seq: str, out_pdb: str | Path) -> dict:
        """Write the model to `out_pdb` with pLDDT (0-100) in the B-factor column, as in AFDB files."""
        with self.torch.no_grad():
            tokens = self.tokenizer([seq], return_tensors="pt", add_special_tokens=False)["input_ids"]
            out = self.model(tokens.to(self.device))
        plddt = out["plddt"][0, :, 1].double().cpu().numpy()    # per residue, at the CA atom
        if plddt.max() <= 1.0:
            plddt *= 100
        lines = []
        for line in self.model.output_to_pdb(out)[0].splitlines():
            if line.startswith("ATOM"):
                line = f"{line[:60]}{plddt[int(line[22:26]) - 1]:6.2f}{line[66:]}"
            lines.append(line)
        Path(out_pdb).write_text("\n".join(lines) + "\n")
        return {"mean_plddt": float(plddt.mean()), "ptm": float(out["ptm"].item())}

    def release(self):
        """Give the GPU memory back (TED's parsers run in their own process and need it)."""
        del self.model
        self.torch.cuda.empty_cache()
