# Operation TED: empirical deletion/shrinking plan

## What is being tested

The PI's proposal is most coherent as a supervised conditional deletion model,
not as a direct copy of SCISOR.  For a protein sequence `x` with TED domain
assignments, define:

- `K`: residues covered by any annotated domain;
- `D`: valid residues not covered by an annotated domain;
- clean endpoint: the ordered subsequence `x[K]`;
- forward state at deletion fraction `rho`: remove a uniformly sampled subset
  of `D`, never a residue in `K`;
- denoising target: a uniform distribution over the non-domain residues still
  present in that shortened state.

Every intermediate state retains its original residue-index map.  It is
physically shortened before ESM-2 encoding, which changes contextual embeddings
as actual inference-time deletion would.  Explicit probability mass at `rho=0`
trains the full-protein start state; the other states cover the path toward the
domain-only endpoint.  A fully deleted state is excluded because it has no next
deletion action.

This is SCISOR-inspired, but it is not claimed to be SCISOR's exact transition
kernel or variational objective.  TED has paired source/target supervision, so
the first scientifically useful experiment is the simpler empirical conditional
path.  Artificial amino-acid insertion and the previous fixed-length masking
objective remain off by default and exist only as controls.

## Architecture

The existing frozen ESM-2 and segmentation Transformer remain intact.  On its
trunk representation `h`, the new path computes:

1. a learned remaining-deletion fraction from masked mean-pooled `h`;
2. budget-conditioned per-residue deletion logits;
3. per-chain standardized deletion evidence;
4. the mean absolute evidence jump across each residue's immediate neighbours;
5. a zero-gated learned projection of these two features back into `h`.

The fused representation feeds both the residue-membership head and the
pairwise same-domain head.  Consequently, the deletion hypothesis can alter
domain boundaries and domain partitioning, instead of being an auxiliary score
that has no route to the final prediction.  The gate starts at zero so an
initialized base checkpoint begins at its original segmentation function.

The CATH head architecture is unchanged.  It pools the fused Stage-A
representation during integrated training.  Frozen CATH vocabularies are shared
between arms; CATH-head weights are not imported from an initialization
checkpoint if its label-to-ID mapping differs.

## Evaluation contract

The CHAINSAW ground truth must be reconstructed from `true_dbounds`, grouping
segments that repeat the same `true_dnames` value.  The raw `true_chopping`
field is retained only for audit because it disagrees with authoritative domain
identity/segment data for a non-trivial subset.

The primary overlap metric is the CHAINSAW chain IoU: Hungarian-matched domain
IoUs weighted by the corresponding ground-truth domain residue counts.  Correct
domain proportion is divided by the number of ground-truth domains.  The former
unweighted metrics remain under explicit `_unweighted` names.  Predictions that
are missing or empty remain empty and score as failures; they are never replaced
with a full-chain domain.  Ranges are clipped to the evaluated chain, and unknown
ground-truth CATH labels do not count as correct `None == None` matches.

Primary model-selection metrics:

1. multi-domain weighted chain IoU;
2. multi-domain correct-domain proportion at IoU >= 0.8;
3. boundary-distance score and domain-count match;
4. single-domain metrics as a regression guard;
5. known-label CATH exact and hierarchical score, reported separately and as a
   joint IoU+CATH success rate.

Deletion AUROC/AUPRC, top-k linker enrichment, boundary-window F1, and budget
error are mechanism diagnostics.  They cannot establish that TED improved if
the final segmentation metrics do not improve.

## Controlled experiment

All arms use one frozen group split and one train-only CATH vocabulary:

- A: architecture/base losses only;
- B: the former full-sequence deletion plus fixed-length linker masking control;
- C: full-sequence deletion plus physically shortened empirical states, with the
  denoiser auxiliary to segmentation;
- D: C plus the learned budget and deletion-feature integration into Stage A.

Each arm runs at seeds 0 and 1 in the first screen.  If D improves the corrected
multi-domain metric without a material single-domain regression, it should be
repeated with at least three additional seeds and tuned only on the frozen
validation split.  Test-set threshold tuning is prohibited.

## Assumptions and decisions to confirm with the PI

The implementation currently makes these explicit assumptions:

1. Every residue outside the supplied TED domains—including terminal tails and
   internal gaps—is a deletion target.  If uncovered regions may contain missed
   domains, labels need an `unknown` state rather than treating them as negative.
2. The first experiment uses deletion evidence to improve one-shot segmentation.
   Actual iterative deletion is diagnostic only; it does not determine the final
   domain calls or stopping rule yet.
3. Discontinuous domains are preserved as one domain identity even though
   removing intervening residues makes their segments adjacent in a shrunk
   sequence.  Evaluation is always projected to original coordinates.
4. `auto` splitting is homology-aware only when a populated homology-cluster
   column exists.  Otherwise it guarantees augmentation-family isolation, not
   remote-homology isolation.  A frozen MMseqs2/UniRef-derived cluster column is
   recommended before publication-level estimates.
5. The benchmark's corrected target and scoring code are the authority for new
   comparisons.  Existing result CSVs were produced under the former contract
   and must not be compared numerically without regeneration.

These decisions do not block the first controlled screen, but items 1, 2, and 4
are the most important questions for the PI before treating the result as a new
TED method rather than a feasibility experiment.

