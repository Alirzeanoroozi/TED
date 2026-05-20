# Two-stage TEDPred (ESM2 + segmentation + hierarchical CATH)

A decomposed replacement for the character-level seq2seq decoder. It addresses
the diagnosed failure modes of the baseline (multi-domain collapse, digit-by-digit
boundaries, free-text CATH) while keeping the same `chopping_star` output format,
so the existing evaluator and figures work unchanged.

## Architecture

```
sequence ──► ESM2 (frozen) ──► per-residue embeddings (B, L, 1280)
                                      │
        Stage A (segmentation, trainable Transformer trunk)
          ├─ residue head     : P(residue ∈ a domain vs linker/NDR)
          └─ pairwise head     : P(residue i, j in the same domain)   [symmetric]
                                      │  cluster (connected-components / spectral)
                                      ▼
                              domains (count is emergent, not generated)
                                      │
        Stage B (CATH, per domain)    ▼
          pool each domain's embedding ──► 4 hierarchical heads C→A→T→H
                                          (each conditioned on its parent level)
                                      │
                                      ▼
            serialize ──► "start-end | C.A.T.H * ..."  (chopping_star)
```

Why this maps onto the Tier-1/Tier-2 recommendations:

| Recommendation | Where |
|---|---|
| 1.1 frozen protein-LM encoder (ESM2-650M) | `esm_backbone.py` (held outside `state_dict`; sequence-only at inference) |
| 1.2 stop generating boundaries as digits | Stage A predicts boundaries over PLM features; no digit decoding |
| 1.3 CATH as a hierarchical classifier | `CathClassifier` — 4 softmax heads, parent-conditioned |
| 1.4 recipe: warmup+cosine, label smoothing, length handling | `train.py` (`LambdaLR`, `label_smoothing`, `LengthBucketedSampler`, chunked ESM embedding) |
| 2.1 segment-then-classify decoupling | CATH head trains on **ground-truth** domains; uses predicted domains at inference |
| 2.2 per-residue / per-pair segmentation, emergent domain count | `SegmentationHead` + `clustering.py` |
| 3.1 class-balanced / focal loss on the CATH head only | `losses.cath_loss` + `--cath_loss_type {ce,focal,class_balanced,cb_focal}` (boundary losses stay on the natural distribution) |
| 3.2 coarse-to-fine CATH levels | `--cath_level_weights` + `--cath_curriculum_steps` (ramp C→A→T→H) |
| 3.3 train/benchmark CATH consistency audit | `benchmark/.../audit_cath_consistency.py` |
| 4.1 decode-strategy ablation (beam) | `CathClassifier.decode` beam over the 4-level hierarchy; `--cath-decode`, `--ablation` |
| 4.2 metrics broken out by domain count | training eval logs `val_subset_single_*` / `val_subset_multi_*`; benchmark prints 1/2/3/4+/2+ buckets |

## Modules

| File | Purpose |
|---|---|
| `cath_vocab.py` | 4 cumulative-prefix CATH vocabularies (C, A, T, H); UNK = `-` |
| `targets.py` | parse `chopping_star` → per-residue assignment, per-domain CATH ids |
| `clustering.py` | co-membership matrix → domains (connected-components or spectral) |
| `serialize.py` | domains + CATH → `chopping_star` string |
| `esm_backbone.py` | frozen ESM2 wrapper, batched + overlap-chunked embeddings |
| `model.py` | `SegmentationHead`, `CathClassifier`, `TwoStageDomainModel` |
| `losses.py` | residue + symmetric pairwise BCE; hierarchical CATH cross-entropy |
| `dataset.py` | `TwoStageDataset`, `collate_fn`, `LengthBucketedSampler` |
| `train.py` | Lightning module + datamodule + W&B + periodic benchmark eval |
| `infer.py` | load checkpoint, predict `chopping_star` strings |

## Train

```bash
pip install fair-esm scikit-learn        # added to requirements.txt
python src/two_stage/train.py \
    --data_parquet_folder data/all_parquet \
    --esm_model_name esm2_t33_650M_UR50D \
    --epochs 30 --batch_size 4 --max_len 1022 \
    --lr 3e-4 --warmup_steps 1000 --label_smoothing 0.1 \
    --save_dir lightning_logs_two_stage \
    --save_path artifacts/two_stage_checkpoint.pt
# or on the cluster:  sbatch slurm/run_train_two_stage.slurm
```

The checkpoint stores only the trainable heads + the CATH vocab; the ~650M ESM2
weights are reloaded from pretrained at load time (kept out of every checkpoint).

For fast iteration use a smaller backbone, e.g. `--esm_model_name esm2_t30_150M_UR50D`.

## Benchmark

```bash
python benchmark/chainsaw_cath1363_from_scratch/run_benchmark_two_stage.py \
    --checkpoint artifacts/two_stage_checkpoint.pt \
    --benchmark-csv benchmark/chainsaw_cath1363_from_scratch/data/processed_with_cath/chainsaw_cath1363_with_cath_labels.csv \
    --output-dir benchmark/chainsaw_cath1363_from_scratch/results_two_stage
```

Produces the same `predictions.csv` / `per_chain_metrics.csv` / `summary.json` /
`figure6a.png` / `figure_cath.png` as the baseline runner (it reuses that runner's
evaluation and plotting code), so results drop straight into the existing comparison.

## CATH long-tail handling (Tier 3) & evaluation (Tier 4)

```bash
# Class-balanced focal loss on the CATH head ONLY (boundaries stay natural):
python src/two_stage/train.py ... \
    --cath_loss_type cb_focal --focal_gamma 2.0 --cb_beta 0.999

# Coarse-to-fine: ramp C->A->T->H over the first 20k steps:
python src/two_stage/train.py ... --cath_curriculum_steps 20000
#   or static emphasis on coarse levels:  --cath_level_weights 1.5,1.25,1.0,0.5

# Audit train-vs-benchmark CATH granularity / indexing / vocab coverage:
python benchmark/chainsaw_cath1363_from_scratch/audit_cath_consistency.py \
    --train-parquet-dir data/all_parquet

# Decode-strategy ablation (greedy vs beam CATH, shared segmentation):
python benchmark/chainsaw_cath1363_from_scratch/run_benchmark_two_stage.py \
    --checkpoint artifacts/two_stage_checkpoint.pt --ablation
```

- **Why CATH head only:** the old WeightedRandomSampler reweighted whole chains, which
  distorted boundary learning (why those runs were worse). Here the long-tail fix lives
  entirely in the CATH cross-entropy (class-balanced weights from train frequencies and/or
  focal); the residue/pairwise segmentation losses see the natural distribution.
- **Beam vs grammar-guided:** the two-stage output is always structurally valid by
  construction (`serialize` emits a parseable string), so the old "grammar-guided" decode is
  unnecessary — `pred_parse_ok` ≈ 1.0. The remaining decode knobs are the **clustering method**
  (segmentation) and **greedy vs beam over the CATH hierarchy** (labels). `--ablation` reports both.
- **Domain-count breakout:** since single-domain was already near Chainsaw parity, training logs
  `val_subset_multi_*` separately and the benchmark prints 1/2/3/4+/2+ buckets, so multi-domain
  progress (the bottleneck) is visible.
- The audit script flags granularity mismatches, index-base mismatches, and low superfamily
  coverage. On the local benchmark CSV it already shows ~20% of CATH1363 domains are unlabeled
  (`-`) and all labeled ones are 4-level, 1-based — so train labels must match (4-level, 1-based).

## Notes / knobs

- **Memory**: Stage A builds an `L×L` pairwise logit map. Training caps length at
  `--max_len` (default 1022 = one ESM2 window); the pairwise *loss* further
  subsamples to `--max_pair_len` residues for long chains. Inference truncates at
  `--max-len` (default 1700) for the same reason. Raise these if GPU memory allows.
- **Clustering**: `--cluster_method connected_components` (default, dependency-free)
  or `spectral` (needs scikit-learn; eigengap-chosen domain count).
- **Domain count** is emergent from clustering — there is no counting-by-generation,
  which was the main multi-domain failure mode of the baseline.
- The old char-level seq2seq (`src/model.py`, `src/train_lightning.py`) is left
  intact as a baseline.
