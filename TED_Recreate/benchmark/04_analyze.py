"""Compare our results with TED's published ones and write tidy tables + a summary (no plotting here).

Inputs (from 02_fetch.py / 03_run.py): proteins.json, chop_afdb.json, chop_esm.json, tm_esm_vs_afdb.json,
labels_ted.json, labels_afdb.json, labels_esm.json, own_cluster.tsv, timings.json
Outputs (results/): ted_domains.tsv (one row per published TED domain, both modes), our_domains.tsv,
proteins.tsv, boundaries.tsv, label_tiers.tsv, label_e2e.tsv, summary.json
"""
import csv, json, re
from collections import Counter
from pathlib import Path

from ted_recreate.chop import parse_chopping

here = Path(__file__).parent
out = here / "results"
out.mkdir(exist_ok=True)
prot = json.load(open(here / "proteins.json"))
MODES = ("afdb", "esm")
chop = {m: {r["name"][:-4] if m == "esm" else r["name"]: r for r in json.load(open(here / f"chop_{m}.json"))["results"]}
        for m in MODES}
failed = {m: json.load(open(here / f"chop_{m}.json"))["failed"] for m in MODES}
tm = json.load(open(here / "tm_esm_vs_afdb.json"))
labels = {k: {r["name"]: r for r in json.load(open(here / f"labels_{k}.json"))} for k in ("ted", "afdb", "esm")}
own_rep = {}
for line in open(here / "own_cluster.tsv"):
    rep, mem, lab, meth = line.rstrip("\n").split("\t")
    own_rep[mem] = rep


def resset(chopping):
    segs = parse_chopping(chopping)[0]
    return {r for a, b in segs for r in range(a, b + 1)}, segs


def iou(a, b):
    return len(a & b) / len(a | b)


def cat3(label):
    return ".".join(label.split(".")[:3])


def outcome(truth_label, truth_level, pred_label):
    """How a predicted label compares with TED's. truth_level in H/T/-."""
    if truth_level == "-":
        return "TED unlabelled: ours unlabelled" if not pred_label else "TED unlabelled: ours labelled"
    if not pred_label:
        return "no label"
    if truth_level == "H":
        if pred_label == truth_label:
            return "same superfamily"
        return "same fold only" if cat3(pred_label) == cat3(truth_label) else "different"
    return "same fold" if cat3(pred_label) == truth_label else "different"      # truth is a T-level label


def agrees(o):
    return o in ("same superfamily", "same fold")


W = {}


def table(name, header):
    fh = open(out / name, "w", newline="")
    w = csv.writer(fh, delimiter="\t")
    w.writerow(header)
    W[name] = (fh, w)
    return w


w_ted = table("ted_domains.tsv", ["chain", "ted_id", "mode", "consensus_level", "nres", "num_segments", "ted_plddt",
                                   "ted_cath", "ted_cath_level", "ted_cath_method", "best_iou", "exact",
                                   "matched_our_id", "our_consensus_level"])
w_our = table("our_domains.tsv", ["chain", "our_id", "mode", "consensus_level", "nres", "num_segments", "plddt",
                                   "best_iou_with_ted", "matched_ted_id"])
w_prot = table("proteins.tsv", ["chain", "nres", "n_ted", "afdb_plddt", "mode", "n_ours", "mean_best_iou", "all_exact",
                                 "residue_agreement", "esm_plddt", "esm_ptm", "tm_esm_afdb", "unidoc_failed"])
w_bnd = table("boundaries.tsv", ["chain", "ted_id", "mode", "kind", "shift"])
matches = {m: {} for m in MODES}      # our domain id -> (ted domain dict, iou)

for chain, v in prot.items():
    ted = []
    for d in v["ted"]:
        s, segs = resset(d["chopping"])
        ted.append({**d, "set": s, "segs": segs})
    for m in MODES:
        r = chop[m].get(chain)
        if r is None:
            continue
        ours = []
        for d in r["domains"]:
            s, segs = resset(d["chopping"])
            ours.append({**d, "set": s, "segs": segs})
        pairs = sorted(((iou(t["set"], o["set"]), i, j) for i, t in enumerate(ted) for j, o in enumerate(ours)), reverse=True)
        t_match, o_match = {}, {}
        for x, i, j in pairs:                      # greedy one-to-one matching by IoU
            if x > 0 and i not in t_match and j not in o_match:
                t_match[i], o_match[j] = (j, x), (i, x)
        best = []
        for i, t in enumerate(ted):
            b = max((iou(t["set"], o["set"]) for o in ours), default=0.0)
            j, x = t_match.get(i, (None, 0.0))
            o = ours[j] if j is not None else None
            best.append(b)
            w_ted.writerow([chain, t["ted_id"], m, t["consensus_level"], t["nres"], t["num_segments"], t["plddt"],
                            t["cath_label"], t["cath_level"], t["cath_method"], f"{b:.4f}",
                            int(o is not None and o["set"] == t["set"]), o["ted_id"] if o else "",
                            o["consensus_level"] if o else ""])
            if o is not None and x >= 0.5:
                ob = sorted({p for a, b_ in o["segs"] for p in (a, b_)})
                for a, b_ in t["segs"]:
                    for kind, p in (("start", a), ("end", b_)):
                        near = min(ob, key=lambda q: abs(q - p))
                        w_bnd.writerow([chain, t["ted_id"], m, kind, near - p])
        for j, o in enumerate(ours):
            b = max((iou(t["set"], o["set"]) for t in ted), default=0.0)
            i, x = o_match.get(j, (None, 0.0))
            w_our.writerow([chain, o["ted_id"], m, o["consensus_level"], o["nres"], o["num_segments"], o["plddt"],
                            f"{b:.4f}", ted[i]["ted_id"] if i is not None else ""])
            if i is not None:
                matches[m][o["ted_id"]] = (ted[i], x)
        t_all = set().union(*[t["set"] for t in ted]) if ted else set()
        o_all = set().union(*[o["set"] for o in ours]) if ours else set()
        agree = sum((res in t_all) == (res in o_all) for res in range(1, v["nres"] + 1)) / v["nres"]
        mm = re.search(r"mean pLDDT ([0-9.]+), pTM ([0-9.]+)", r["structure_source"])
        w_prot.writerow([chain, v["nres"], len(ted), f"{v['afdb_mean_plddt']:.2f}", m, len(ours),
                         f"{sum(best)/len(best):.4f}", int(all(x == 1.0 for x in best) and len(ours) == len(ted)),
                         f"{agree:.4f}", mm.group(1) if mm else "", mm.group(2) if mm else "",
                         tm.get(chain, {}).get("tm", "") if m == "esm" else "",
                         int(r["methods"].get("unidoc") == "NO_SS")])

# ---- label tiers on TED's own published domains ----
w_tier = table("label_tiers.tsv", ["ted_id", "ted_cath", "ted_level", "ted_method", "tier", "pred", "pred_level", "outcome", "detail"])
for chain, v in prot.items():
    for d in v["ted"]:
        L = labels["ted"].get(d["ted_id"])
        if L is None:
            continue
        truth, lev = d["cath_label"], d["cath_level"]
        ex = [e for e in L["ted_exact"] if e["level"] != "-"]
        preds = {"exact lookup": (ex[0]["cath_label"] if ex else None, ex[0]["level"] if ex else "-", f"{len(L['ted_exact'])} identical TED domains")}
        st = L["structure"] or {}
        preds["structure"] = (st.get("cath_label"), st.get("level", "-"), st.get("method", ""))
        sq = L["sequence_transfer"]
        preds["sequence"] = (sq["cath_label"] if sq else None, sq["level"] if sq else "-",
                             "joins cluster" if sq else "no cluster joined")
        rep = own_rep.get(d["ted_id"])
        others = [h for h in L["sequence_hits"] if h["ted_rep"] != rep]
        joins = [h for h in others if h["joins_ted_cluster"]]
        if joins:
            h = max(joins, key=lambda h: h["bits"])
            preds["sequence (own cluster removed)"] = (None if h["cluster_label"] == "-" else h["cluster_label"],
                                                       h["cluster_level"], f"id={h['fident']:.2f}")
        else:
            preds["sequence (own cluster removed)"] = (None, "-", "no other cluster joined")
        lab_hits = [h for h in others if h["cluster_label"] != "-"]
        preds["nearest labelled homolog (own cluster removed)"] = (
            (lab_hits[0]["cluster_label"], lab_hits[0]["cluster_level"], f"id={lab_hits[0]['fident']:.2f}") if lab_hits
            else (None, "-", "no labelled homolog"))
        for tier, (p, pl, detail) in preds.items():
            w_tier.writerow([d["ted_id"], truth, lev, d["cath_method"], tier, p or "", pl, outcome(truth, lev, p), detail])

# ---- end to end: our domains' final label vs the TED domain they correspond to ----
w_e2e = table("label_e2e.tsv", ["our_id", "mode", "matched_ted_id", "iou", "ted_cath", "ted_level", "final", "final_level",
                                 "source", "outcome", "structure_pred", "sequence_pred", "novel_final", "novel_source",
                                 "novel_outcome"])


def as_if_new(L, rep):
    """Label the function would give if the protein were not in TED: no exact lookup, and the matched TED
    domain's own sequence cluster removed from the MMseqs2 database. Structure first, then cluster transfer."""
    st = L["structure"] or {}
    if st.get("cath_label"):
        return st["cath_label"], st["method"]
    joins = [h for h in L["sequence_hits"] if h["ted_rep"] != rep and h["joins_ted_cluster"]]
    if joins:
        h = max(joins, key=lambda h: h["bits"])
        if h["cluster_label"] != "-":
            return h["cluster_label"], "mmseqs-transfer"
    return None, "none"


for m in MODES:
    for oid, L in labels[m].items():
        if oid not in matches[m]:
            nl, ns = as_if_new(L, None)
            w_e2e.writerow([oid, m, "", 0, "", "", L["cath_label"] or "", L["level"], L["source"], "no TED counterpart", "", "",
                            nl or "", ns, "no TED counterpart"])
            continue
        t, x = matches[m][oid]
        st, sq = L["structure"] or {}, L["sequence_transfer"] or {}
        nl, ns = as_if_new(L, own_rep.get(t["ted_id"]))
        w_e2e.writerow([oid, m, t["ted_id"], f"{x:.4f}", t["cath_label"], t["cath_level"], L["cath_label"] or "", L["level"],
                        L["source"], outcome(t["cath_label"], t["cath_level"], L["cath_label"]),
                        st.get("cath_label") or "", sq.get("cath_label") or "", nl or "", ns,
                        outcome(t["cath_label"], t["cath_level"], nl)])
for fh, _ in W.values():
    fh.close()

# ---- summary numbers ----
def rows(name):
    return list(csv.DictReader(open(out / name), delimiter="\t"))


S = {"n_proteins": len(prot), "n_ted_domains": sum(len(v["ted"]) for v in prot.values()),
     "failed": {m: len(failed[m]) for m in MODES}, "timings": json.load(open(here / "timings.json"))}
td, pr, bd = rows("ted_domains.tsv"), rows("proteins.tsv"), rows("boundaries.tsv")
for m in MODES:
    t = [r for r in td if r["mode"] == m]
    p = [r for r in pr if r["mode"] == m]
    b = [abs(int(r["shift"])) for r in bd if r["mode"] == m]
    o = [r for r in rows("our_domains.tsv") if r["mode"] == m]
    S[m] = {
        "proteins": len(p), "ted_domains": len(t), "our_domains": len(o),
        "proteins_all_domains_exact": sum(int(r["all_exact"]) for r in p) / len(p),
        "proteins_same_domain_count": sum(r["n_ted"] == r["n_ours"] for r in p) / len(p),
        "ted_domains_exact": sum(int(r["exact"]) for r in t) / len(t),
        **{f"ted_domains_iou>={x}": sum(float(r["best_iou"]) >= x for r in t) / len(t) for x in (0.9, 0.8, 0.5)},
        "our_domains_with_ted_counterpart_iou>=0.8": sum(float(r["best_iou_with_ted"]) >= 0.8 for r in o) / max(1, len(o)),
        "our_domains_without_counterpart_iou<0.5": sum(float(r["best_iou_with_ted"]) < 0.5 for r in o) / max(1, len(o)),
        "mean_residue_agreement": sum(float(r["residue_agreement"]) for r in p) / len(p),
        **{f"boundaries_within_{k}": sum(x <= k for x in b) / len(b) for k in (0, 1, 2, 5, 8, 20)},
        "unidoc_failed": sum(int(r["unidoc_failed"]) for r in p),
    }
tiers = rows("label_tiers.tsv")
S["label_tiers"] = {}
for tier in dict.fromkeys(r["tier"] for r in tiers):
    tr = [r for r in tiers if r["tier"] == tier]
    lab = [r for r in tr if r["ted_level"] != "-"]
    unl = [r for r in tr if r["ted_level"] == "-"]
    S["label_tiers"][tier] = {
        "ted_labelled": len(lab), "outcomes": dict(Counter(r["outcome"] for r in lab)),
        "agree_at_ted_level": sum(agrees(r["outcome"]) for r in lab) / len(lab),
        "same_fold_or_better": sum(r["outcome"] in ("same superfamily", "same fold", "same fold only") for r in lab) / len(lab),
        "precision_when_labelled": (sum(agrees(r["outcome"]) for r in lab) / max(1, sum(r["outcome"] != "no label" for r in lab))),
        "ted_H": {"n": sum(r["ted_level"] == "H" for r in lab), "same_superfamily": sum(r["outcome"] == "same superfamily" for r in lab)},
        "ted_T": {"n": sum(r["ted_level"] == "T" for r in lab), "same_fold": sum(r["outcome"] == "same fold" for r in lab)},
        "ted_unlabelled": len(unl), "ted_unlabelled_also_unlabelled": sum(r["outcome"].endswith("ours unlabelled") for r in unl)}
e2e = rows("label_e2e.tsv")
S["label_e2e"] = {}
for m in MODES:
    e = [r for r in e2e if r["mode"] == m]
    same = [r for r in e if r["matched_ted_id"] and float(r["iou"]) >= 0.8]
    lab = [r for r in same if r["ted_level"] != "-"]
    # per published TED domain: was it recovered (IoU>=0.8) AND given TED's label?
    by_ted = {r["matched_ted_id"]: r for r in same}
    t_lab = [r for r in td if r["mode"] == m and r["ted_cath_level"] != "-"]
    S["label_e2e"][m] = {
        "our_domains": len(e), "sources": dict(Counter(r["source"] for r in e)),
        "matched_iou>=0.8": len(same), "of_which_ted_labelled": len(lab),
        "label_agrees_at_ted_level": sum(agrees(r["outcome"]) for r in lab) / max(1, len(lab)),
        "outcomes_ted_labelled": dict(Counter(r["outcome"] for r in lab)),
        "outcomes_ted_unlabelled": dict(Counter(r["outcome"] for r in same if r["ted_level"] == "-")),
        "ted_labelled_domains": len(t_lab),
        "ted_labelled_recovered_and_same_label": sum(1 for r in t_lab if r["ted_id"] in by_ted and agrees(by_ted[r["ted_id"]]["outcome"])) / max(1, len(t_lab)),
        "ted_labelled_recovered": sum(1 for r in t_lab if r["ted_id"] in by_ted) / max(1, len(t_lab)),
        # "as if new to TED": exact lookup off, own cluster removed
        "new_label_agrees_at_ted_level": sum(agrees(r["novel_outcome"]) for r in lab) / max(1, len(lab)),
        "new_outcomes_ted_labelled": dict(Counter(r["novel_outcome"] for r in lab)),
        "new_sources": dict(Counter(r["novel_source"] for r in e)),
        "new_ted_labelled_recovered_and_same_label": sum(1 for r in t_lab if r["ted_id"] in by_ted and agrees(by_ted[r["ted_id"]]["novel_outcome"])) / max(1, len(t_lab)),
        "new_ted_labelled_recovered_same_fold_or_better": sum(1 for r in t_lab if r["ted_id"] in by_ted and by_ted[r["ted_id"]]["novel_outcome"] in ("same superfamily", "same fold", "same fold only")) / max(1, len(t_lab)),
    }
json.dump(S, open(out / "summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))
