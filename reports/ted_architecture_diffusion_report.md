# TED Model Architecture and SCISOR-Inspired Diffusion Report

Date: 2026-08-04

This report summarizes the TED model lineage in this repository: the original TED
sequence-to-text Transformer, the newer two-stage TED model, and the current
SCISOR-inspired deletion-diffusion extension. It also explains which ideas were
adapted from `references/8_A_Diffusion_Model_to_Shrink_.pdf`, including the math
used in the paper and the local TED implementation.

## 1. Project Goal

TED predicts protein domain annotations from an amino-acid sequence. The target
format is a `chopping_star` string:

```text
start-end | C.A.T.H * start-end | C.A.T.H * ...
```

For each chain, the model should learn:

- Which residues belong to domains versus non-domain/linker regions.
- How residues are grouped into one or more domains.
- The CATH label for each predicted domain.
- A serializable, evaluator-compatible `chopping_star` output.

The central failure of the original implementation was that domain boundaries and
CATH labels were generated as free text. This made multi-domain outputs fragile:
the model could collapse multiple domains into one, emit invalid grammar, and
make digit-level boundary mistakes.

## 2. Original TED Implementation: Character-Level Seq2Seq

Main files:

- `src/model.py`
- `src/train_lightning.py`
- `src/data/dataset_.py`
- `src/tokenizer_.py`

### 2.1 Input and Output

Input:

- Raw amino-acid sequence string.
- Character-tokenized with a learned `TextTokenizer`.
- Special tokens: `<pad>`, `<sos>`, `<eos>`, `<unk>`.
- Training default maximum source length: `max_src_len=1024`.

Output:

- Raw `chopping_star` string.
- Character-tokenized target string.
- Training default maximum target length: `max_tgt_len=256`.

Training sample:

```text
sequence -> "MKT..."
target   -> "5-104 | 3.40.50.300 * 130-245 | 2.60.40.10"
```

The model learns next-character prediction over the target string.

### 2.2 Architecture

Class: `TextToTextTransformer`

Architecture:

- Source embedding: `src_vocab_size x d_model`.
- Target embedding: `tgt_vocab_size x d_model`.
- Shared sinusoidal positional encoding.
- Transformer encoder:
  - `num_encoder_layers=4`
  - `d_model=256`
  - `nhead=8`
  - `dim_feedforward=1024`
  - activation `gelu`
- Transformer decoder:
  - `num_decoder_layers=4`
  - same hidden dimensions as encoder
  - causal target mask
- Output projection: `d_model -> tgt_vocab_size`.

### 2.3 Parameter Count

From `artifacts/transformer_checkpoint.pt` and
`artifacts/transformer_checkpoint_grammar.pt`:

| Component | Parameters |
|---|---:|
| Source embedding | 6,144 |
| Target embedding | 5,120 |
| Transformer encoder | 3,159,040 |
| Transformer decoder | 4,213,760 |
| Output projection | 5,140 |
| **Total trainable** | **7,389,204** |

Checkpoint state dict count: `7,651,860`.

The difference between the instantiated module count and the saved state count
comes from checkpoint-specific buffers/state such as positional encodings.

### 2.4 Objective

The old model uses token-level cross entropy:

```math
L_text = - \sum_t \log p_\theta(y_t \mid y_{<t}, x)
```

where `x` is the sequence and `y` is the serialized `chopping_star`.

### 2.5 Limitations

The original model has no explicit concept of:

- A residue being in-domain versus non-domain.
- Pairwise residue co-membership.
- Domain count.
- CATH hierarchy.
- Boundary uncertainty.
- Deletion/linker signal.

Everything is compressed into next-character generation. This is why valid
syntax can improve with grammar-guided decoding, but semantic domain grouping
can still fail.

## 3. Two-Stage TED Before Diffusion

Main files:

- `src/two_stage/model.py`
- `src/two_stage/train.py`
- `src/two_stage/dataset.py`
- `src/two_stage/targets.py`
- `src/two_stage/clustering.py`
- `src/two_stage/serialize.py`

Checkpoint:

- `artifacts/two_stage_checkpoint.pt`

### 3.1 Input and Output

Input:

- Raw amino-acid sequence.
- Embedded by frozen ESM2.
- In the previous trained checkpoint, the backbone is `esm2_t30_150M_UR50D`.
- ESM2 has about `148,140,154` frozen parameters.
- ESM2 weights are not saved in TED checkpoints; only trainable TED heads are saved.

Intermediate target:

- `assign`: per-residue integer assignment.
  - `0`: linker/non-domain residue.
  - `1..K`: domain id.
- Ground-truth domains parsed from `chopping_star`.
- Per-domain CATH ids at four hierarchy levels.

Output:

- Predicted per-residue domain assignment.
- Predicted CATH code for each domain.
- Serialized back to `chopping_star`.

### 3.2 Architecture

The two-stage model decomposes the problem:

```text
sequence
  -> frozen ESM2 embeddings, shape (B, L, 640) for esm2_t30_150M_UR50D
  -> Stage A segmentation head
  -> predicted domains by clustering
  -> Stage B CATH classifier
  -> serialized chopping_star
```

Note: The README diagram mentions `(B, L, 1280)` for the larger ESM2-650M
variant. The actual checkpoints discussed here use ESM2-150M with embedding
dimension `640`.

### 3.3 Stage A: Segmentation Head

Class: `SegmentationHead`

Hyperparameters from checkpoint:

- `d_model=512`
- `nhead=8`
- `num_seg_layers=4`
- `dim_feedforward=2048`
- `pair_dim=128`
- `dropout=0.1`

Submodules:

- `input_proj`: projects ESM embedding dimension to `d_model`.
- Transformer encoder trunk over residues.
- Residue head:
  - predicts `P(residue in a domain)`.
- Pairwise co-membership head:
  - projects trunk features into query/key vectors.
  - computes symmetric pair logits:

```math
z_{ij}^{pair}
= \frac{1}{2}
\left(
  \frac{q_i^\top k_j}{\sqrt{d_{pair}}}
  +
  \frac{q_j^\top k_i}{\sqrt{d_{pair}}}
\right)
 + b
```

The pairwise matrix and residue probabilities are clustered into domains by
connected components or spectral clustering. Domain count is therefore emergent,
not generated as text.

### 3.4 Stage B: Hierarchical CATH Classifier

Class: `CathClassifier`

Input feature per domain:

```text
mean pooled ESM features || mean pooled Stage-A trunk features
```

For ESM2-150M:

```text
feat_dim = 640 + 512 = 1152
```

The classifier has:

- Shared MLP trunk.
- Four CATH heads: C, A, T, H.
- Parent-conditioned decoding:
  - A head is conditioned on chosen C.
  - T head is conditioned on chosen A.
  - H head is conditioned on chosen T.
- Greedy or beam decoding over the four-level hierarchy.

### 3.5 Parameter Count Before Diffusion

From `artifacts/two_stage_checkpoint.pt`:

| Component | Parameters |
|---|---:|
| Stage A segmentation head | 13,069,570 |
| Stage B CATH head | 3,471,101 |
| **Saved trainable params before diffusion** | **16,540,671** |
| Frozen ESM2-150M backbone | 148,140,154 |

CATH vocabulary sizes in that checkpoint:

```text
[5, 41, 1081, 3286]
```

Meaning:

- 5 class-level labels.
- 41 architecture-level cumulative prefixes.
- 1081 topology-level cumulative prefixes.
- 3286 homologous-superfamily-level cumulative prefixes.

### 3.6 Losses Before Diffusion

Residue loss:

```math
y_i^{res} = 1[a_i > 0]
```

```math
L_res =
\operatorname{BCEWithLogits}(z_i^{res}, y_i^{res})
```

Pairwise loss:

```math
y_{ij}^{pair} = 1[a_i = a_j \land a_i > 0]
```

```math
L_pair =
\operatorname{BCEWithLogits}(z_{ij}^{pair}, y_{ij}^{pair})
```

CATH loss:

```math
L_CATH =
\sum_{\ell \in \{C,A,T,H\}} w_\ell
\operatorname{CE}(z_\ell, y_\ell)
```

The local implementation supports focal and class-balanced CATH loss:

```math
L_{focal} = - (1 - p_t)^\gamma \log p_t
```

and class-balanced weights from training-set CATH frequencies. These weights are
applied only to the CATH head, not to segmentation, to avoid distorting boundary
learning.

Total pre-diffusion objective:

```math
L =
\lambda_{res} L_{res}
+ \lambda_{pair} L_{pair}
+ \lambda_{CATH} L_{CATH}
```

## 4. Current TED: Diffusion-Augmented Two-Stage Model

Main added/modified files:

- `src/two_stage/model.py`
- `src/two_stage/diffusion.py`
- `src/two_stage/losses.py`
- `src/two_stage/train.py`
- `src/two_stage/infer.py`
- `benchmark/chainsaw_cath1363_from_scratch/run_benchmark_two_stage.py`
- `slurm/run_train_two_stage_diffusion*.slurm`
- `slurm/run_benchmark_two_stage_diffusion.slurm`

Completed bounded checkpoint:

- `artifacts/two_stage_diffusion_bounded_checkpoint.pt`

Active long-run checkpoint:

- `lightning_logs_two_stage_diffusion/last.ckpt`

### 4.1 New Deletion Diffusion Head

Class: `DeletionDiffusionHead`

Purpose:

- Predict a residue-level distribution over deletion candidates.
- In TED, deletion candidates mean residues that should be removed to recover
  domain-only structure from a full chain.
- In v1, this is supervised by known non-domain residues and artificial inserted
  residues.

Input:

- Stage-A trunk representation:

```text
trunk_rep: (B, L, 512)
```

- Residue mask:

```text
mask: (B, L)
```

- Deletion budget:

```text
S_b = number of residues to delete for sequence b
```

Budget conditioning features:

```math
c_b = [\log(1 + S_b), S_b / L_b]
```

The budget embedding is:

```math
e_b = f_\theta(c_b) \in \mathbb{R}^{512}
```

The deletion head predicts:

```math
z_i^{del}
= h_\theta(\operatorname{LayerNorm}(r_i + e_b))
```

where `r_i` is the Stage-A trunk vector for residue `i`.

The deletion distribution is:

```math
q_\theta(i \mid X, S)
=
\operatorname{softmax}_i(z_i^{del})
```

### 4.2 Current Parameter Count

From `artifacts/two_stage_diffusion_bounded_checkpoint.pt`:

| Component | Parameters |
|---|---:|
| Stage A segmentation head | 13,069,570 |
| New deletion diffusion head | 199,041 |
| Stage B CATH head | 3,559,187 |
| **Saved trainable params, current bounded model** | **16,827,798** |
| Frozen ESM2-150M backbone | 148,140,154 |

CATH vocabulary sizes in the bounded checkpoint:

```text
[5, 41, 1105, 3412]
```

The current checkpoint has a slightly larger CATH head than
`two_stage_checkpoint.pt` because the bounded training run used a different
train/validation split ratio and therefore rebuilt the CATH vocabulary with
more observed rare labels.

### 4.3 New Losses

#### 4.3.1 Supervised Real Non-Domain Deletion Loss

For real TED samples, the target delete set is:

```math
D_b = \{i : a_{b,i}=0 \land mask_{b,i}=1\}
```

The target distribution is uniform over non-domain residues:

```math
p^*_{b,i}
=
\begin{cases}
1 / |D_b|, & i \in D_b \\
0, & \text{otherwise}
\end{cases}
```

The loss is cross entropy, equivalent to KL up to the target entropy constant:

```math
L_{del-real}
=
- \frac{1}{B'}
\sum_{b: |D_b|>0}
\frac{1}{|D_b|}
\sum_{i \in D_b}
\log q_\theta(i \mid X_b, S_b)
```

where:

```math
S_b = |D_b|
```

#### 4.3.2 Artificial Insertion Deletion Loss

For each training sequence:

1. Extract the domain-only clean sequence:

```math
X_0 = X[\{i : a_i > 0\}]
```

2. Sample a noising time:

```math
t \sim Uniform(0, t_{max})
```

3. Compute SCISOR-style alpha:

```math
\alpha(t) = (1 - t)^{1 / \gamma}
```

In local TED code:

```text
t_max = 0.9
gamma / alpha_rate = 1.1
```

4. Sample the total number of insertions:

```math
M_t \sim NegativeBinomial(L + 1, 1 - \alpha(t))
```

The PyTorch parameterization uses `total_count=L+1` and `probs=1-alpha`.
The sample is clamped by:

```text
max_insertions = 128
max_noised_len = 1022
```

and forced to at least one insertion for useful auxiliary supervision.

5. Uniformly choose one of `L+1` insertion sites for each inserted residue.

6. Draw inserted amino acids uniformly from the 20 canonical amino acids:

```text
ACDEFGHIKLMNPQRSTVWY
```

7. Train the deletion head to put probability mass on the known inserted
positions.

If `I_b` is the set of inserted positions:

```math
L_{del-artificial}
=
- \frac{1}{B'}
\sum_{b: |I_b|>0}
\frac{1}{|I_b|}
\sum_{i \in I_b}
\log q_\theta(i \mid X_{t,b}, M_{t,b})
```

#### 4.3.3 Current Total Loss

The current training objective is:

```math
L =
\lambda_{res}L_{res}
+ \lambda_{pair}L_{pair}
+ \lambda_{del}L_{del-real}
+ \lambda_{art}L_{del-artificial}
+ \lambda_{CATH}L_{CATH}
```

The bounded run used:

```text
lambda_res  = 1.0
lambda_pair = 1.0
lambda_del  = 0.25
lambda_art  = 0.10
lambda_CATH = 1.0
```

### 4.4 Fused Inference Ablation

The current implementation supports two inference modes:

1. Auxiliary only:
   - Train with deletion losses.
   - Do not alter segmentation inference.

2. Fused inference:
   - Use deletion logits to adjust residue domain logits before clustering.

Deletion evidence is computed by a masked log-softmax centered by chain mean:

```math
e_i =
\log q_\theta(i \mid X,S)
-
\frac{1}{L}
\sum_j \log q_\theta(j \mid X,S)
```

Residue logits are adjusted as:

```math
z_i^{res,fused}
=
z_i^{res}
-
\alpha_{fuse} e_i
```

In the benchmark:

```text
alpha_fuse = 1.0
```

Interpretation:

- If a residue has higher-than-average deletion evidence, fused inference makes
  it less likely to be treated as a domain residue.
- If a residue has lower-than-average deletion evidence, it is relatively
  protected.

### 4.5 Iterative Diagnostic Shrink Mode

The current model has a diagnostic shrink mode:

```text
diagnostic_shrink(sequence, deletion_fraction=0.1)
```

Algorithm:

1. Start with full sequence `X`.
2. Set deletion budget `S`.
3. Predict deletion logits.
4. Delete the highest-logit residue, or sample with temperature if enabled.
5. Record:
   - step
   - current index
   - original index
   - residue
   - deletion logit
   - residue domain probability
   - current length
6. Repeat until the desired number of residues is removed.

This mode is diagnostic. It is not yet used as the production decoder.

### 4.6 Events in the Current Implementation

The word "event" appears in two relevant senses.

Diffusion event:

- In SCISOR, one event is one insertion in the forward process.
- In the reverse process, one event corresponds to deleting one residue.
- The event count is `M_t`: the number of inserted residues present at time `t`.
- In TED artificial noising, `M_t` is sampled per sequence from the
  negative-binomial schedule and then clamped to:

```text
1 <= M_t <= min(128, max_noised_len - clean_len)
```

- In real TED supervised deletion, the event budget is:

```math
S_b = |\{i : a_{b,i}=0\}|
```

meaning the number of known non-domain residues in sequence `b`.

Training/checkpoint/evaluation event:

- Bounded v1 run: `12000` training steps.
- Periodic validation benchmark callback: every `2000` steps, so six logged
  benchmark events at steps `2000, 4000, 6000, 8000, 10000, 12000`.
- Step checkpoint cadence: every `1000` training steps via `last.ckpt`.
- Full CHAINSAW benchmark after bounded training: one evaluation event over
  `1356` CATH1363 chains, run in two ablation modes.

## 5. SCISOR Paper: Ideas and Math Adapted to TED

Reference:

- `references/8_A_Diffusion_Model_to_Shrink_.pdf`

The paper introduces SCISOR: Sequence Contraction with InSertion-Only noising
pRocess. The key idea is to train a model to delete by defining a forward process
that only inserts random residues. The reverse process must therefore delete
residues.

### 5.1 SCISOR Forward Process

Let:

```math
X_0 = X_0^{(1)}, \dots, X_0^{(L)}
```

be a natural protein sequence. There are `L+1` insertion sites: before the first
residue, between every pair of residues, and after the last residue.

SCISOR uses a pure-birth process with rate:

```math
\beta(t)
```

and insertion letter distribution:

```math
\pi
```

The probability of no insertion at a site by time `t` is:

```math
\alpha(t)
=
\exp\left(
  - \int_0^t \beta(s)\,ds
\right)
```

The number of inserted residues by time `t` is:

```math
M_t \sim NegativeBinomial(L+1, \alpha(t))
```

The expected noised length is:

```math
E[|X_t|] = E[L + M_t] = \frac{L+1}{\alpha(t)} - 1
```

Then the `M_t` insertions are distributed uniformly into `L+1` bins, and each
inserted residue is sampled from:

```math
Y \sim Cat(\pi)
```

### 5.2 SCISOR Reverse Process

The reverse model is a denoiser:

```math
q_\theta(X_{t_{M-1}} \mid X_{t_M}, M)
```

Equivalently, it predicts a distribution over which residue to delete:

```math
q_\theta(prev(X_t) \mid X_t, M)
```

Here `M` is the number of insertions still present. Conditioning on `M` is
important because the model can choose different deletion policies depending on
how many deletions remain.

### 5.3 SCISOR ELBO Training Term

The paper derives an upper bound on negative log likelihood:

```math
-\log q_\theta(X_0 \mid L)
\le
E_{M_1}
KL(p(X_1 \mid X_0,M_1) \| q(X_1 \mid L+M_1))
+
E_{t,X_t,M_t}
\frac{M_t\beta(t)}{1-\alpha(t)}
KL(
  p(prev(X_t) \mid X_0,X_t,M_t)
  \|
  q_\theta(prev(X_t) \mid X_t,M_t)
)
```

The first term says long noised sequences become close to random sequences drawn
from `Cat(pi)`. The second term is the practical denoising loss: train the model
to identify which residue was inserted last.

### 5.4 SCISOR Deletion Posterior

SCISOR computes the true deletion target by summing over all alignments from
the clean sequence `X0` to the noised sequence `Xt`.

Let:

```math
ali(X,Y)
```

be the number of alignments of sequence `X` to sequence `Y`. If deleting one
letter from `Xt` gives `prev(Xt)`, the posterior is:

```math
p(prev(X_t) \mid X_0,X_t,M_t)
=
\frac{
  ali(X_0, prev(X_t))
}{
  M_t \cdot ali(X_0, X_t)
}
```

Intuition:

- A residue that often aligns to the clean sequence is probably not noise.
- A residue that is a gap in many alignments is more likely to be an insertion
  and therefore a good deletion candidate.

### 5.5 SCISOR Schedule

The paper uses:

```math
\beta(t) = \frac{\gamma}{1 - t_{max}t}
```

so:

```math
\alpha(t)
=
(1 - t_{max}t)^{\gamma/t_{max}}
```

The paper reports using approximately:

```text
gamma = 1.1
t_max = 0.9
```

In the local TED implementation, this is simplified to:

```math
\alpha(t) = (1 - t)^{1/\gamma}
```

with:

```text
t sampled from [0, 0.9]
gamma/alpha_rate = 1.1
```

This preserves the same qualitative behavior: as `t` grows, `alpha(t)`
decreases and the negative-binomial insertion count increases.

### 5.6 SCISOR Architecture Idea

SCISOR uses ESM2-style protein-language-model features and modifies the model to
condition on `M`. The paper describes FiLM-style conditioning:

```math
a_{\ell,d}
\mapsto
(1 + A_{\theta,\ell,d}(M))a_{\ell,d}
+ B_{\theta,\ell,d}(M)
```

The TED adaptation does not insert FiLM layers into every ESM block. Instead,
because TED keeps ESM frozen and has a trainable Stage-A trunk, it conditions
the deletion head by adding a learned budget embedding to trunk features:

```math
r_i \mapsto LayerNorm(r_i + f_\theta([\log(1+S), S/L]))
```

This is a local, lightweight analogue of SCISOR's count conditioning.

## 6. What Was Adapted From SCISOR and How

| SCISOR idea | TED adaptation |
|---|---|
| Learn deletion by reversing insertion noise | Domain-only residues are treated as clean; non-domain/inserted residues are deletion targets. |
| Pure-birth insertion process | `diffusion.py` samples random inserted amino acids into domain-only sequences. |
| Negative-binomial insertion count | Implemented with `NegativeBinomial(total_count=L+1, probs=1-alpha)`. |
| Condition denoiser on number of insertions/deletions `M` | `DeletionDiffusionHead` conditions on `[log1p(S), S/L]`. |
| Denoiser predicts distribution over next deletion | TED deletion head outputs one logit per residue and uses masked softmax. |
| Posterior deletion target from inserted/noisy residues | For artificial noising, the exact inserted mask is used as the target. For real TED, known non-domain residues are used. |
| Iterative shrinking | Implemented as diagnostic shrink mode, not production decoding. |
| Deletion probability as structural/functional importance signal | TED uses deletion scores for non-domain AUROC/AUPRC, boundary F1, linker enrichment, and visualizations. |

## 7. What Was Not Fully Implemented From SCISOR

The current TED version is a v1 supervised diffusion adaptation. It does not yet
implement the full SCISOR generative framework.

Not implemented:

- Full continuous-time ELBO weighting:

```math
\frac{M_t\beta(t)}{1-\alpha(t)}
```

- Dynamic-programming alignment posterior:

```math
\frac{ali(X_0, prev(X_t))}{M_t ali(X_0,X_t)}
```

- Corrector steps with repeated noise/de-noise.
- Production iterative decoding for final domain segmentation.
- Sampling multiple deletions per denoiser step.
- Function-conditioned guidance.
- Inserting FiLM layers throughout ESM2.

The current implementation deliberately keeps the existing two-stage TED
architecture intact and adds deletion diffusion as an auxiliary and diagnostic
signal.

## 8. What Was Implemented in the Last Step

### 8.1 Deletion Head in `TwoStageDomainModel`

Added:

- `DeletionDiffusionHead`
- `deletion_logits_from_embeddings`
- deletion logits in `forward_train`
- auxiliary-return inference path
- diagnostic shrink mode

### 8.2 Supervised Deletion Loss

Added:

- `deletion_mask_ce_loss`
- `deletion_ce_loss`

Real TED target:

```text
assign == 0
```

Artificial target:

```text
inserted_mask == True
```

### 8.3 SCISOR-Style Artificial Insertion Noising

Added in `src/two_stage/diffusion.py`:

- domain-only clean-sequence extraction.
- negative-binomial insertion schedule.
- random amino-acid insertion.
- inserted-mask target generation.
- padded deletion masks.

### 8.4 Fusion Ablation

Added:

- auxiliary-only inference.
- fused inference by subtracting deletion evidence from residue logits.

Benchmark output includes both:

- `summary_aux_only.json`
- `summary_fused.json`
- `figure_deletion_aux_only.png`
- `figure_deletion_fused.png`

### 8.5 Diagnostic Shrinking

Added:

- iterative deletion trace generation.
- `diagnostic_shrink_traces.csv`
- `diagnostic_shrink_summary.json`
- `diagnostic_shrink_examples.png`

### 8.6 Benchmark Metrics

Added deletion metrics:

- Non-domain AUROC.
- Non-domain AUPRC.
- Boundary F1.
- Top-k linker enrichment.

Existing CHAINSAW-style metrics retained:

- IoU.
- Correct proportion.
- Boundary distance score.
- NDO.
- Domain-count match.
- CATH correctness.
- CATH level score.
- Multi-domain split.

## 9. Bounded v1 Training and Benchmark Results

Bounded training run:

- Slurm job: `1465709`
- W&B run: `https://wandb.ai/esevinc22-koc-university-/ted_with_logs/runs/c7x0tkem`
- Checkpoint: `artifacts/two_stage_diffusion_bounded_checkpoint.pt`
- Steps: `12000`
- Runtime: about 32 minutes

Benchmark run:

- Slurm job: `1465744`
- Results directory:
  `benchmark/chainsaw_cath1363_from_scratch/results_two_stage_diffusion_bounded`
- Number of benchmark chains: `1356`

### 9.1 CHAINSAW Benchmark Comparison

| Model / mode | Overall IoU | Correct proportion | Boundary distance score | Domain count match | Multi-domain IoU | Multi-domain correct |
|---|---:|---:|---:|---:|---:|---:|
| Current, auxiliary only | 0.638 | 0.518 | 0.303 | 0.695 | 0.463 | 0.313 |
| Current, fused inference | 0.609 | 0.410 | 0.184 | 0.706 | 0.452 | 0.247 |
| CHAINSAW reference, 1+ domains | 0.880 | 0.786 | 0.789 | n/a | n/a | n/a |
| CHAINSAW reference, 2+ domains | 0.832 | 0.731 | 0.679 | n/a | n/a | n/a |

Interpretation:

- The deletion auxiliary signal is useful diagnostically.
- The naive fused inference rule with `alpha=1.0` hurt segmentation quality in
  this bounded v1 run.
- The model remains well below CHAINSAW on the full benchmark, especially
  boundary distance and multi-domain performance.

### 9.2 Deletion Metrics

| Group | Non-domain AUROC | Non-domain AUPRC | Boundary F1 | Top-k linker enrichment |
|---|---:|---:|---:|---:|
| Overall | 0.777 | 0.453 | 0.193 | 5.893 |
| Single-domain | 0.928 | 0.539 | 0.223 | 10.943 |
| Multi-domain | 0.711 | 0.414 | 0.185 | 3.661 |

Interpretation:

- The deletion head has learned a meaningful non-domain signal.
- The signal is much stronger for single-domain chains than multi-domain chains.
- Boundary F1 is still low, so deletion probability is not yet a precise boundary
  detector.

### 9.3 Diagnostic Shrink Summary

From `diagnostic_shrink_summary.json`:

```text
n_targets = 8
n_deleted = 202
deleted_true_non_domain_fraction = 0.0545
deleted_near_boundary_fraction = 0.1040
```

This diagnostic used only a small sample. It is useful for qualitative
visualization, but not yet a production shrinker.

## 10. Current Job State

The bounded v1 train and benchmark are complete.

The full long training job remains separate:

- Training job: `1459130`
- Current checkpoint: `lightning_logs_two_stage_diffusion/last.ckpt`
- Dependent full benchmark job: `1466066`
- Full final artifact expected:
  `artifacts/two_stage_diffusion_checkpoint.pt`

The full benchmark is queued to run after the long job succeeds.

## 11. Key Takeaways

1. The original TED model was a pure text-generation model with about 7.39M
   trainable parameters and no explicit residue/domain structure.

2. The two-stage model replaced free-text generation with explicit segmentation
   and hierarchical CATH classification, using frozen ESM2 features and about
   16.54M trainable head parameters before diffusion.

3. The current diffusion-augmented model adds a 199K-parameter deletion head,
   bringing the bounded checkpoint to about 16.83M trainable parameters, while
   still relying on the same frozen ESM2-150M backbone.

4. The SCISOR paper inspired the deletion-by-reversing-insertions framing, the
   negative-binomial insertion schedule, budget-conditioned deletion logits, and
   iterative shrink diagnostics.

5. TED does not yet implement full SCISOR. It uses a supervised local adaptation:
   known non-domain residues and artificial insertions define deletion targets.

6. The deletion head already produces a measurable non-domain signal
   (`AUROC=0.777` overall), but simple fused inference currently hurts
   segmentation and needs calibration or a better integration strategy.

## 12. Recommended Next Technical Steps

1. Keep the deletion head as an auxiliary signal for now; do not default to fused
   inference with `alpha=1.0`.

2. Sweep fusion strength:

```text
alpha_fuse in {0.05, 0.1, 0.2, 0.5}
```

3. Add a boundary-aware loss that uses deletion peaks near true boundaries/linkers
   without suppressing true short domains.

4. Replace the uniform non-domain target with a structured target:

```math
p^*_i \propto
\lambda_{linker} 1[i \in linker]
+ \lambda_{boundary} 1[i \in boundary window]
```

5. Move from diagnostic shrink to controlled decoding only after deletion
   calibration improves on multi-domain chains.

6. For closer SCISOR fidelity, implement alignment-posterior targets for
   artificial insertion examples rather than only the exact inserted mask.

## 13. V2 Implementation Addendum

This addendum records the follow-up implementation from the v2 deletion-diffusion
spec.

### 13.1 Reproducibility and Model Selection Guardrails

Training now exposes:

```text
--seed
--shuffle_seed
```

`src/two_stage/train.py` calls:

```python
pl.seed_everything(args.seed, workers=True)
```

before datamodule/model construction. The pairwise subsampling generator and the
diffusion/noising generator are derived from this seed:

```text
pair generator seed  = seed + 101
noise generator seed = seed + 202
```

The training split order remains controlled independently by `--shuffle_seed`.
Both values are saved through the Lightning hyperparameters and W&B config.

Checkpoint selection no longer monitors `eval_loss`. The best checkpoint now
monitors the held-out validation-subset multi-domain segmentation metric:

```text
monitor = val_subset_multi_iou_chain
mode    = max
```

The CATH1363 Chainsaw benchmark remains a final/monitoring benchmark and should
not be used to choose checkpoints.

### 13.2 Empirical Linker-Dropout Forward Process

The new v2 forward noising helper is:

```text
src/two_stage/diffusion.py::make_linker_dropout_batch
```

For a sequence with residue-domain assignment `a_i`, valid mask `m_i`, and true
linker/non-domain set

```math
D = \{ i : a_i = 0 \land m_i = 1 \},
```

the empirical noising process samples:

```math
\rho \sim \mathrm{Uniform}(0, 1)
```

and marks

```math
k = \lfloor \rho |D| \rfloor
```

true linker residues as already deleted. With `R = D \setminus D_{deleted}`,
the remaining deletion budget is:

```math
S_t = |R|.
```

The training target is the uniform posterior over the remaining true linker
positions:

```math
q(i \mid x_t, S_t) =
\begin{cases}
1 / |R|, & i \in R \\
0,       & \text{otherwise}
\end{cases}
```

and the deletion head is trained with:

```math
\mathcal{L}_{linker-dropout}
= -\frac{1}{|R|}\sum_{i \in R}
\log p_\theta(i \mid x_t, S_t).
```

This is the cheap TED-style implementation: tensor length is not physically
changed. Already-deleted linker residues are removed from the deletion softmax
mask, and the existing ESM/trunk features are reused.

The loss is controlled by:

```text
--linker_dropout_loss_weight  # default 0.25
```

and is logged as:

```text
train_linker_dropout_deletion_loss
eval_linker_dropout_deletion_loss
```

The validation value is zero by design because this stochastic objective is
training-only.

### 13.3 Deletion Head Feature Ablation

The deletion head now supports:

```text
--deletion_head_input trunk
--deletion_head_input esm_trunk
```

`trunk` reproduces the previous v1 behavior:

```math
h_i = z_i
```

where `z_i` is the segmentation trunk representation.

`esm_trunk` concatenates raw frozen ESM features and the segmentation trunk
features:

```math
h_i = W [e_i ; z_i],
```

then applies the same budget-conditioned residual normalization:

```math
\tilde h_i =
\mathrm{LayerNorm}(h_i + c(S_t, L_t)).
```

The budget embedding still uses:

```math
c(S_t, L_t) =
\mathrm{MLP}([\log(1 + S_t), S_t/L_t]).
```

This addresses the v1 diagnosis that the deletion head saw nearly the same
features as the residue classifier and could collapse into an opposite-signed
copy of the residue head.

### 13.4 Benchmark Budget Diagnostics

`run_benchmark_two_stage.py` now records:

```text
true_deletion_budget
pred_deletion_budget
budget_abs_error
budget_pearson
```

The predicted budget is computed from the unfused residue probability:

```math
\hat S = \sum_i (1 - p_\theta(domain_i)).
```

The true budget is the number of CATH1363 residues not covered by a reference
domain. Groupwise Pearson correlation between `S` and `S_hat` is included in
`summary.json` under `deletion_metrics`.

### 13.5 Task-1 Ablation Infrastructure

Two Slurm array scripts were added:

```text
slurm/run_train_two_stage_diffusion_ablation_v2.slurm
slurm/run_benchmark_two_stage_diffusion_ablation_v2.slurm
```

The six training jobs implement the requested three-arm, two-seed ablation:

| arm | seeds | deletion loss | artificial insertion loss | linker dropout loss |
| --- | --- | --- | --- | --- |
| A no diffusion | 0, 1 | 0.00 | 0.00 | 0.00 |
| B real deletion only | 0, 1 | 0.25 | 0.00 | 0.00 |
| C real + artificial current | 0, 1 | 0.25 | 0.10 | 0.00 |

The benchmark array evaluates all six checkpoints with:

```text
--deletion-fusion-alpha 0.0
```

and writes:

```text
benchmark/chainsaw_cath1363_from_scratch/results_ablation_A_seed0/
benchmark/chainsaw_cath1363_from_scratch/results_ablation_A_seed1/
...
```

The comparison table utility was added at:

```text
benchmark/chainsaw_cath1363_from_scratch/compare_runs.py
```

It accepts one or more `summary.json` paths, emits a Markdown table over
overall/single/multi metrics, includes Gen1 and Chainsaw reference rows, and
flags overlapping confidence intervals between supplied runs.
