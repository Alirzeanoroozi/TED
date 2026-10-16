# TED_Recreate

Chop a protein into domains and give each domain a CATH label the way TED did (The Encyclopedia of Domains;
Lau et al., Science 2024). The chopping runs TED's own released pipeline, so a result for a new protein means what
a TED entry means. We use it to make TED-style labels for new sequences and to score our sequence-based models
against TED.

```python
from ted_recreate import ted_chop, cath_label_chopped

protein = ted_chop(uniprot="P00533")              # or ted_chop("MKT...") or ted_chop(pdb="model.pdb")
labels = cath_label_chopped([protein])
for domain, label in zip(protein.domains, labels):
    print(domain.ted_id, domain.chopping, domain.consensus_level, label.cath_label, label.name_of_label)
```

```
AF-P00533-F1-model_v4_TED01 31-207 high 3.80.20.20 Receptor L-domain
AF-P00533-F1-model_v4_TED02 703-793 medium 3.30.200.20 Phosphorylase Kinase; domain 1
AF-P00533-F1-model_v4_TED03 796-991 medium 1.10.510.10 Transferase(Phosphotransferase) domain 1
```

## Domains: `ted_chop`

| Input | Structure that gets chopped |
|---|---|
| `ted_chop(uniprot="P00533")` | the AlphaFold DB model, which is what TED chopped |
| `ted_chop("MKT...")` | an ESMFold model of the sequence |
| `ted_chop(pdb="model.pdb")` | your own model (first chain) |

`ted_chop_batch([...])` does many proteins in one go. Each result has:

- `domains`: the TED domains, each with `ted_id`, `chopping` (residue ranges such as `11-41_290-389`),
  `consensus_level`, `sequence`, `plddt`.
- `methods`: what Merizo, Chainsaw and UniDoc each proposed.
- `residue_labels()`: the domain number of every residue, 0 outside domains. Handy as a training target.

What runs, in TED's released code (`vendor/ted-tools`, unmodified):

1. **Merizo** on the model, **UniDoc** on the residues Merizo kept, **Chainsaw** on the model.
2. Each parser's output is filtered: segments under 5 residues and domains under 25 are dropped.
3. **Consensus.** Domains from different parsers are linked when they overlap with IoU ≥ 0.7. A group of three is
   `high`, of two `medium`, and only those become TED domains. The range kept is what the group shares.

Three things to know:

- **Sequence input cannot match TED exactly.** TED chopped AlphaFold models and ESMFold is a different predictor.
  For a UniProt protein, use `uniprot=`. ESMFold's output also shifts slightly between PyTorch versions, so
  sequence-input results can differ a little from one install to the next. The other two inputs do not.
- **Boundaries are in TED's numbering**, which sits one residue after what the parsers agreed on. That shift is in
  TED's code and in its database. `chopping_parser_frame` gives the unshifted ranges.
- **UniDoc version.** The UniDoc package ships two builds of the program. TED's published UniDoc results match the
  2022 build (identical for 97 of 106 proteins, against 38 for the newer one), so `setup.sh` installs that one.

## CATH labels: `cath_label`

```python
from ted_recreate import cath_label, cath_label_chopped

label = cath_label("DOMAINSEQUENCE...")       # from the sequence alone
label = cath_label(pdb="domain.pdb")          # also uses the structure
labels = cath_label_chopped(results)          # every domain of ted_chop results
label.cath_label, label.level, label.source, label.name_of_label
```

Three sources are tried in turn, and the first that gives a label decides. `level` is `H` for a superfamily label
(`3.40.50.300`) and `T` for a fold label (`3.40.50`).

| Source | What it does |
|---|---|
| `exact` | The sequence is one of TED's 365M domains: return TED's published label. |
| `structure` | TED's method, run here: Foldseek against CATH 4.3 with TED's thresholds, then Foldclass and TM-align for what Foldseek leaves unlabelled. Needs the domain's structure. |
| `sequence` | MMseqs2 against the 120.7M representatives of TED's sequence clusters. A domain that meets TED's clustering rule against a representative (≥ 50% identity, length ratio ≥ 0.9) gets that cluster's label. Weaker homologs are reported as `remote_hint` only. |

Two things differ from TED because TED did not publish them: the CATH search set (we use CATH 4.3 S40 for TED's
"SSG5" list) and the exact Foldseek build (we use release 8).

The `sequence` source searches a 15 GB database, so something has to hold it in memory. Inside a GPU job the search
loads it for that one run. To label from anywhere, start the MMseqs server once and `cath_label` finds it:

```bash
sbatch scripts/mmseqs_server.sbatch                                          # CPU: 200 GB RAM, no GPU
sbatch --gres=gpu:tesla_a100:1 --mem=64G scripts/mmseqs_server.sbatch gpu    # or GPU
```

A batch of 8 domains takes about 17 s either way. Stop the server with `scancel` when you are done.

## Command line

```bash
python -m ted_recreate chop  --uniprot P00533              # domains from the AlphaFold DB model
python -m ted_recreate label --fasta proteins.fa --out labels.json     # fold, chop, label every domain
python -m ted_recreate cath  --fasta domains.fa            # labels for domain sequences
sbatch scripts/gpu.sbatch -m ted_recreate label --fasta proteins.fa --out labels.json      # the same on a GPU node
```

Folding and chopping need a GPU for anything beyond a few small proteins: about 2 s per protein from an existing
model and 12 s from sequence on one A100.

## How close is it to TED?

Tested on 600 random TED proteins with 1,037 published domains. Full report with figures:
[benchmark/REPORT.md](benchmark/REPORT.md).

| | From the AlphaFold model | From sequence (ESMFold) |
|---|---|---|
| TED domains reproduced residue for residue | 98.0% | 21.7% |
| TED domains matched at overlap ≥ 0.8 | 98.7% | 79.8% |
| Boundaries within ±8 residues | 99.6% | 84.2% |
| TED-labelled domains recovered and given TED's label | 99.2% | 81.1% |
| The same, scored as if the protein were new to TED | 93.9% | 79.2% |

It reproduces TED-100. TED's 40M "redundant" domains were made with a different consensus function and are not
reproduced.

## Install

Linux, Python 3.10, and about 400 GB of disk for the lookups.

```bash
git clone https://github.com/erkmenerken/TED_Recreate && cd TED_Recreate
python3 -m venv .venv && source .venv/bin/activate && pip install -e .     # PyTorch and the rest: several GB
bash setup.sh                             # TED's code, the three parsers, Foldseek, MMseqs2: 5 min, 1 GB
bash scripts/download_data.sh             # TED's and CATH's published tables: hours, 80 GB
sbatch scripts/build_lookup_dbs.sbatch    # the lookups cath_label searches: about 2 h, 290 GB
```

The ESMFold weights (8 GB) download themselves into `models/` the first time a sequence is folded. The Slurm
headers in `scripts/*.sbatch` name our cluster's account, partition and GPU type; change them for yours. Submit jobs
from the repository root.

`ted_chop` works after the first three commands. `cath_label` needs the last two.

## Layout

```
ted_recreate/    the package: chop.py (ted_chop), classify.py (cath_label), structure.py (AlphaFold DB, ESMFold),
                 server.py (MMseqs server), __main__.py (command line), config.py (paths)
setup.sh         downloads the third-party code and programs into vendor/ and tools/
scripts/         data download, lookup builders, Slurm wrappers
benchmark/       the 600-protein test against TED: scripts 01 to 06, REPORT.md, figures/
```
