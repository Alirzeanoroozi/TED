"""Write results/tables.md, the "Numbers" section of REPORT.md, from results/summary.json."""
import csv, json
from pathlib import Path

here = Path(__file__).parent
S = json.load(open(here / "results" / "summary.json"))
rows = lambda n: list(csv.DictReader(open(here / "results" / n), delimiter="\t"))
pr = rows("proteins.tsv")
p = lambda x: f"{100 * x:.1f}%"
A, E = S["afdb"], S["esm"]
out = []
w = out.append

w("## Numbers\n")
w(f"Benchmark set: **{S['n_proteins']} TED-100 proteins** with **{S['n_ted_domains']:,} published TED domains**. "
  f"Proteins that failed: {S['failed']['afdb']} (AlphaFold-model mode), {S['failed']['esm']} (sequence-only mode).\n")
w("### Domain chopping\n")
w("| | AlphaFold model (TED's input) | Sequence only (ESMFold) |\n|---|---|---|")
for label, key in [("Proteins with every domain identical to TED", "proteins_all_domains_exact"),
                   ("Proteins with the same number of domains", "proteins_same_domain_count"),
                   ("TED domains reproduced exactly", "ted_domains_exact"),
                   ("TED domains matched at overlap ≥ 0.9", "ted_domains_iou>=0.9"),
                   ("TED domains matched at overlap ≥ 0.8", "ted_domains_iou>=0.8"),
                   ("TED domains matched at overlap ≥ 0.5", "ted_domains_iou>=0.5"),
                   ("Our domains with a TED counterpart (overlap ≥ 0.8)", "our_domains_with_ted_counterpart_iou>=0.8"),
                   ("Our domains with no TED counterpart (overlap < 0.5)", "our_domains_without_counterpart_iou<0.5"),
                   ("Residues with the same in-domain / not-in-domain status", "mean_residue_agreement"),
                   ("Boundaries identical to TED's", "boundaries_within_0"),
                   ("Boundaries within ±2 residues", "boundaries_within_2"),
                   ("Boundaries within ±8 residues", "boundaries_within_8"),
                   ("Boundaries within ±20 residues", "boundaries_within_20")]:
    w(f"| {label} | {p(A[key])} | {p(E[key])} |")
w(f"| Domains produced | {A['our_domains']:,} | {E['our_domains']:,} |")
w(f"| Proteins where UniDoc failed | {A['unidoc_failed']} | {E['unidoc_failed']} |\n")

e = [r for r in pr if r["mode"] == "esm" and r["tm_esm_afdb"] not in ("", "None")]
tmv = sorted(float(r["tm_esm_afdb"]) for r in e)
plv = sorted(float(r["esm_plddt"]) for r in e if r["esm_plddt"])
w(f"ESMFold models vs AlphaFold models: median TM-score {tmv[len(tmv)//2]:.2f}; "
  f"{p(sum(x >= 0.9 for x in tmv)/len(tmv))} of proteins at TM ≥ 0.9, {p(sum(x < 0.5 for x in tmv)/len(tmv))} below 0.5. "
  f"Median ESMFold mean pLDDT {plv[len(plv)//2]:.0f}.\n")

w("### CATH labels, each route on TED's own published domains\n")
w("| Route | Same label as TED | Same fold or better | Different label | No label | Correct when it gives a label | TED-unlabelled domains we label |\n|---|---|---|---|---|---|---|")
for tier, t in S["label_tiers"].items():
    n, o = t["ted_labelled"], t["outcomes"]
    w(f"| {tier} | {p(t['agree_at_ted_level'])} | {p(t['same_fold_or_better'])} | {p(o.get('different', 0)/n)} | "
      f"{p(o.get('no label', 0)/n)} | {p(t['precision_when_labelled'])} | "
      f"{p(1 - t['ted_unlabelled_also_unlabelled']/max(1, t['ted_unlabelled']))} |")
t0 = next(iter(S["label_tiers"].values()))
w(f"\nBase: {t0['ted_labelled']} TED-labelled domains ({t0['ted_H']['n']} with a superfamily, {t0['ted_T']['n']} with a fold only) "
  f"and {t0['ted_unlabelled']} TED-unlabelled domains.\n")
w("| Route | TED superfamily labels reproduced | TED fold-only labels reproduced |\n|---|---|---|")
for tier, t in S["label_tiers"].items():
    w(f"| {tier} | {t['ted_H']['same_superfamily']}/{t['ted_H']['n']} ({p(t['ted_H']['same_superfamily']/max(1,t['ted_H']['n']))}) | "
      f"{t['ted_T']['same_fold']}/{t['ted_T']['n']} ({p(t['ted_T']['same_fold']/max(1,t['ted_T']['n']))}) |")

w("\n### End to end (chop, then label)\n")
w("| | AlphaFold model (TED's input) | Sequence only (ESMFold) |\n|---|---|---|")
LA, LE = S["label_e2e"]["afdb"], S["label_e2e"]["esm"]
w(f"| TED-labelled domains recovered (overlap ≥ 0.8) | {p(LA['ted_labelled_recovered'])} | {p(LE['ted_labelled_recovered'])} |")
w(f"| ... recovered **and** given TED's label | {p(LA['ted_labelled_recovered_and_same_label'])} | {p(LE['ted_labelled_recovered_and_same_label'])} |")
w(f"| Label agreement among recovered TED-labelled domains | {p(LA['label_agrees_at_ted_level'])} | {p(LE['label_agrees_at_ted_level'])} |")
w(f"| As if new to TED*: recovered **and** given TED's label | {p(LA['new_ted_labelled_recovered_and_same_label'])} | {p(LE['new_ted_labelled_recovered_and_same_label'])} |")
w(f"| As if new to TED*: recovered and same fold or better | {p(LA['new_ted_labelled_recovered_same_fold_or_better'])} | {p(LE['new_ted_labelled_recovered_same_fold_or_better'])} |")
w(f"| As if new to TED*: label agreement among recovered domains | {p(LA['new_label_agrees_at_ted_level'])} | {p(LE['new_label_agrees_at_ted_level'])} |")
for src, name in [("ted-exact", "exact lookup"), ("foldseek", "Foldseek"), ("foldclass", "Foldclass"),
                  ("mmseqs-transfer", "sequence cluster"), ("none", "no label")]:
    w(f"| Our domains decided by: {name} | {p(LA['sources'].get(src, 0)/LA['our_domains'])} | {p(LE['sources'].get(src, 0)/LE['our_domains'])} |")

w("\n\\* exact lookup switched off and the domain's own TED sequence cluster removed from the MMseqs2 database.")
T = S["timings"]
w("\n### Run time (one A100, 8 CPU cores)\n")
w("| Stage | Total | Per protein or domain |\n|---|---|---|")
if "chop_afdb_seconds" in T:
    w(f"| Chop from AlphaFold models ({T['chop_afdb_n']} proteins) | {T['chop_afdb_seconds']/60:.0f} min | {T['chop_afdb_seconds']/T['chop_afdb_n']:.1f} s per protein |")
if "chop_esm_seconds" in T:
    w(f"| ESMFold + chop from sequence ({T['chop_esm_n']} proteins) | {T['chop_esm_seconds']/60:.0f} min | {T['chop_esm_seconds']/T['chop_esm_n']:.1f} s per protein |")
for k, nm in [("ted", "TED's published domains"), ("afdb", "our domains, AlphaFold models"), ("esm", "our domains, ESMFold models")]:
    if f"label_{k}_seconds" in T:
        w(f"| CATH labels, all three routes: {nm} ({T[f'label_{k}_n']} domains) | {T[f'label_{k}_seconds']/60:.1f} min | {T[f'label_{k}_seconds']/T[f'label_{k}_n']:.2f} s per domain |")
(here / "results" / "tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
