# TED Deletion Signal Audit

Data path: `data/all_parquet`
Max training length: `1022`
Rows scanned: `1,614,842`
Parse failures: `0`

## Overall

| View | Chains | Residues | Non-domain residues | Weighted non-domain fraction | Chains with any non-domain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 1,614,842 | 449,071,569 | 111,226,595 | 0.2477 | 0.9827 |
| Capped | 1,614,842 | 447,677,615 | 110,574,831 | 0.2470 | 0.9827 |

## Verdict

- Real TED labels contain enough non-domain residue signal to train a supervised deletion auxiliary head, with artificial noising as a robustness add-on.

## Notes

- `non_domain_residues` are residues not covered by any parsed TED domain interval.
- `capped` reflects the current two-stage training view after sequence truncation.
- See the CSV files in this directory for group-by-domain-count, source, and length splits.
