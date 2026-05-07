# CHAINSAW CATH1363 Benchmark From Scratch

This folder is for preparing the TEDPred benchmark dataset without using the
old copied files under `benchmark/results`.

The target benchmark is the CHAINSAW CATH1363 test set used in Figure 6(a) of
the CHAINSAW paper. Their official GitHub folder says:

- train/test splits: `chainsaw_cath1363_train_test_splits.json`
- ground truth domains, CHAINSAW predictions, and metrics:
  `chainsaw_model_v3_on_cath1363_test.csv`

Source:

```text
https://github.com/JudeWells/chainsaw/tree/main/data_and_benchmarking
```

## Step 1: Download Official CHAINSAW Files

From the TED repo root:

```bash
cd /scratch/erkmenerken22/latestTED/TED
python3 benchmark/chainsaw_cath1363_from_scratch/download_official_chainsaw.py
```

This writes:

```text
benchmark/chainsaw_cath1363_from_scratch/data/raw/README.md
benchmark/chainsaw_cath1363_from_scratch/data/raw/chainsaw_cath1363_train_test_splits.json
benchmark/chainsaw_cath1363_from_scratch/data/raw/chainsaw_model_v3_on_cath1363_test.csv
benchmark/chainsaw_cath1363_from_scratch/data/raw/manifest.json
```

## Step 2: Build TEDPred Input Dataset

Run:

```bash
cd /scratch/erkmenerken22/latestTED/TED
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate ted
python benchmark/chainsaw_cath1363_from_scratch/prepare_tedpred_benchmark.py --write_parquet
```

This downloads the needed PDB files from RCSB into:

```text
benchmark/chainsaw_cath1363_from_scratch/data/structures/pdb/
```

Then it writes TEDPred-ready files:

```text
benchmark/chainsaw_cath1363_from_scratch/data/processed/chainsaw_cath1363_tedpred.csv
benchmark/chainsaw_cath1363_from_scratch/data/processed/chainsaw_cath1363_tedpred.parquet
benchmark/chainsaw_cath1363_from_scratch/data/processed/chainsaw_cath1363_excluded.csv
benchmark/chainsaw_cath1363_from_scratch/data/processed/chainsaw_cath1363_summary.json
```

## Format Conversion

CHAINSAW uses zero-based inclusive ranges:

```text
0-259|260-442
```

TEDPred uses one-based inclusive `chopping_star` labels:

```text
1-260 | - * 261-443 | -
```

The CATH class is `-` because this benchmark is for domain boundary comparison,
not CATH classification.

## Why Extract Sequence From PDB Files?

CHAINSAW says the benchmark predictions assume zero-indexed PDB files with
consecutive residue indices. For TEDPred, the safest matching sequence is
therefore the amino-acid sequence extracted from the same PDB chain residue
order, not a pre-existing FASTA helper table.
