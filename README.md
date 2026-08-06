# TED / TED-Pred

TED-Pred is a sequence-only protein-domain annotation project.  The current
`operation_ted` line uses a frozen ESM-2 protein language model followed by a
trainable segmentation Transformer.  Stage A predicts domain-residue membership
and pairwise same-domain probabilities, which are clustered into an emergent
number of domains.  Stage B pools each predicted/teacher-forced domain and
classifies its hierarchical CATH C/A/T/H label.

The current research hypothesis is documented in
[`reports/operation_ted_shrink_plan.md`](reports/operation_ted_shrink_plan.md).
The implementation treats domain annotation as a supervised deletion path:

```text
full protein -> delete annotated non-domain residues -> domain-residue sequence
```

Unlike the earlier fixed-length masking loss, intermediate sequences are
physically shortened and re-embedded by ESM-2.  The deletion denoiser produces
two features for Stage A: standardized deletion evidence and its local jump
magnitude.  A learned budget head estimates the remaining non-domain fraction,
so the integrated model does not need the true deletion count at inference.

## Cluster workflow

Prepare the corrected CHAINSAW benchmark and the frozen experiment metadata:

```bash
BENCH_PREP_JOB=$(sbatch --parsable slurm/run_prepare_chainsaw_corrected.slurm)
DATA_PREP_JOB=$(sbatch --parsable slurm/run_prepare_operation_ted.slurm)
```

If a true homology-cluster column has been added to the parquet data, set
`SPLIT_GROUP_COL` to that column and `REQUIRE_HOMOLOGY_GROUP=1`.  Without it,
the automatic fallback still keeps each original protein with all of its
augmented crops, but is not a complete sequence-homology split.

Launch the eight-job training array only after metadata preparation succeeds,
then benchmark only after the entire training array succeeds:

```bash
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${DATA_PREP_JOB} \
  slurm/run_train_operation_ted_shrink.slurm)
sbatch --dependency=afterok:${TRAIN_JOB}:${BENCH_PREP_JOB} \
  slurm/run_benchmark_operation_ted_shrink.slurm
```

The training array compares four arms at seeds 0 and 1:

| Arm | Deletion training | Physical re-embedding | Deletion features affect Stage A | Learned budget |
|---|---|---:|---:|---:|
| A | none | no | no | no |
| B | full-state + legacy fixed mask | no | no | no |
| C | full-state + empirical intermediate shrink | yes | no | no |
| D | full-state + empirical intermediate shrink | yes | yes | yes |

Primary selection is based on corrected CHAINSAW multi-domain segmentation,
not deletion-head AUROC alone.  Run the prepared tests in the cluster environment
before training:

```bash
pytest -q tests
```
