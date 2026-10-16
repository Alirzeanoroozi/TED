"""Figures for the benchmark (reads results/*.tsv written by 04_analyze.py). Output: figures/*.png and *.pdf

Colour jobs (validated with the dataviz skill's validate_palette.js, light surface):
  the two input modes are categorical: blue = AlphaFold model (TED's input), orange = sequence only (ESMFold)
  label outcomes: dark blue = same label as TED, mid blue = same fold only, orange = different, gray = no label
"""
import csv, json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ted_recreate.chop import parse_chopping

here = Path(__file__).parent
res, figdir = here / "results", here / "figures"
figdir.mkdir(exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DBLUE, MBLUE, GRAY, LGRAY = "#184f95", "#5598e7", "#c3c2b7", "#e8e7e2"
CAT8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODE = {"afdb": ("AlphaFold model (TED's input)", BLUE), "esm": ("Sequence only (ESMFold model)", ORANGE)}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "grid.linestyle": "-", "legend.frameon": False, "legend.fontsize": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


def rows(name):
    return list(csv.DictReader(open(res / name), delimiter="\t"))


def head(fig, title, subtitle, y=0.98):
    fig.text(0.012, y, title, fontsize=13.5, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.012, y - 0.062 * (4.2 / fig.get_figheight()), subtitle, fontsize=9.8, color=INK2, va="top", ha="left")


def ygrid(ax):
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"{name}.{ext}", dpi=220)
    plt.close(fig)
    print("wrote", name)


S = json.load(open(res / "summary.json"))
td, od, pr, bd = rows("ted_domains.tsv"), rows("our_domains.tsv"), rows("proteins.tsv"), rows("boundaries.tsv")
tiers, e2e = rows("label_tiers.tsv"), rows("label_e2e.tsv")
NP, ND = S["n_proteins"], S["n_ted_domains"]
pct = lambda x: f"{100 * x:.0f}%"

# ------------------------------------------------------------------------------------------------------
# 1. How well are TED's published domains recovered?  (survival curve of best IoU)
# ------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.6))
fig.subplots_adjust(left=0.085, right=0.83, top=0.82, bottom=0.13)
xs = [0.5 + 0.005 * i for i in range(101)]
for m, (name, col) in MODE.items():
    v = [float(r["best_iou"]) for r in td if r["mode"] == m]
    ys = [100 * sum(b >= x - 1e-9 for b in v) / len(v) for x in xs]
    ax.plot(xs, ys, color=col, label=name)
    ax.plot([1.0], [ys[-1]], "o", ms=8, color=col, mec=SURFACE, mew=2, clip_on=False, zorder=5)
    ax.annotate(f"{ys[-1]:.0f}% identical", (1.0, ys[-1]), xytext=(9, 0), textcoords="offset points", va="center",
                fontsize=9.5, color=INK, annotation_clip=False)
    i8 = xs.index(0.8) if 0.8 in xs else 60
    ax.annotate(f"{ys[60]:.0f}%", (0.8, ys[60]), xytext=(0, 8 if m == "afdb" else -15), textcoords="offset points",
                ha="center", fontsize=9, color=INK2)
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0, 102)
ax.set_xlabel("Overlap required with the TED domain (intersection over union)")
ax.set_ylabel("TED domains recovered (%)")
ygrid(ax)
ax.legend(loc="lower left")
head(fig, "TED's domains: recovered from its structures, mostly from sequence",
     f"Share of {ND:,} published TED domains (in {NP} proteins) matched by one of our domains at each overlap level")
save(fig, "fig1_domain_recovery")

# ------------------------------------------------------------------------------------------------------
# 2. Number of domains per protein: ours minus TED
# ------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.4))
fig.subplots_adjust(left=0.085, right=0.98, top=0.82, bottom=0.15)
cats = ["2 or more\nfewer", "1 fewer", "same number", "1 more", "2 or more\nextra"]
w = 0.2
for k, (m, (name, col)) in enumerate(MODE.items()):
    d = [int(r["n_ours"]) - int(r["n_ted"]) for r in pr if r["mode"] == m]
    c = [sum(x <= -2 for x in d), d.count(-1), d.count(0), d.count(1), sum(x >= 2 for x in d)]
    ys = [100 * x / len(d) for x in c]
    xs = [i + (k - 0.5) * (w + 0.03) for i in range(5)]
    ax.bar(xs, ys, width=w, color=col, label=name)
    for x, y in zip(xs, ys):
        if y >= 0.5:
            ax.text(x, y + 1.2, f"{y:.0f}%", ha="center", va="bottom", fontsize=9, color=INK2)
ax.set_xticks(range(5), cats)
ax.set_ylabel("Proteins (%)")
ax.set_ylim(0, 105)
ygrid(ax)
ax.legend(loc="upper left")
head(fig, "The number of domains per protein usually matches TED's",
     f"Our number of domains minus TED's, per protein ({NP} proteins)")
save(fig, "fig2_domain_count")

# ------------------------------------------------------------------------------------------------------
# 3. Boundary accuracy
# ------------------------------------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.5), gridspec_kw={"width_ratios": [1, 1]})
fig.subplots_adjust(left=0.07, right=0.985, top=0.79, bottom=0.14, wspace=0.22)
for m, (name, col) in MODE.items():
    b = [abs(int(r["shift"])) for r in bd if r["mode"] == m]
    ys = [100 * sum(x <= k for x in b) / len(b) for k in range(0, 21)]
    a1.plot(range(0, 21), ys, color=col, label=name)
    for k in (0, 8):
        a1.plot([k], [ys[k]], "o", ms=8, color=col, mec=SURFACE, mew=2, zorder=5, clip_on=False)
        a1.annotate(f"{ys[k]:.0f}%", (k, ys[k]), xytext=(9, -15), textcoords="offset points", fontsize=9, color=INK2)
a1.set_xlim(0, 20)
a1.set_ylim(0, 102)
a1.set_xticks([0, 2, 4, 8, 12, 16, 20])
a1.set_xlabel("Distance from TED's boundary (residues)")
a1.set_ylabel("Boundaries within that distance (%)")
a1.set_title("Cumulative boundary error")
ygrid(a1)
a1.legend(loc="lower right")
sh = [int(r["shift"]) for r in bd if r["mode"] == "esm"]
cnt = Counter(max(-15, min(15, x)) for x in sh)
ks = list(range(-15, 16))
a2.bar(ks, [100 * cnt[k] / len(sh) for k in ks], width=0.78, color=ORANGE)
a2.set_xticks([-15, -10, -5, 0, 5, 10, 15], ["≤-15", "-10", "-5", "0", "5", "10", "≥15"])
a2.set_xlabel("Our boundary minus TED's (residues)")
a2.set_ylabel("Boundaries (%)")
a2.set_title("Sequence-only mode: signed shift")
ygrid(a2)
nb = {m: sum(r["mode"] == m for r in bd) for m in MODE}
head(fig, "Boundaries from sequence alone usually land within a few residues of TED's",
     f"Segment start and end points of matched domains (overlap ≥ 0.5): {nb['afdb']:,} boundaries from AlphaFold models, "
     f"{nb['esm']:,} from ESMFold models", y=0.975)
save(fig, "fig3_boundaries")

# ------------------------------------------------------------------------------------------------------
# 4. What drives agreement in sequence-only mode?
# ------------------------------------------------------------------------------------------------------
pinfo = {r["chain"]: r for r in pr if r["mode"] == "esm"}
t_esm = [r for r in td if r["mode"] == "esm" and r["chain"] in pinfo]


def binned(ax, key, edges, names, xlabel):
    vals = []
    for lo, hi in edges:
        sel = [r for r in t_esm if pinfo[r["chain"]][key] not in ("", "None") and lo <= float(pinfo[r["chain"]][key]) < hi]
        vals.append((100 * sum(float(r["best_iou"]) >= 0.8 for r in sel) / len(sel) if sel else 0, len(sel)))
    ax.bar(range(len(edges)), [v for v, _ in vals], width=0.3, color=ORANGE)
    for i, (v, n) in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(edges)), [f"{nm}\nn = {n}" for nm, (_, n) in zip(names, vals)])
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 108)
    ygrid(ax)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.5), sharey=True)
fig.subplots_adjust(left=0.07, right=0.985, top=0.79, bottom=0.21, wspace=0.08)
binned(a1, "tm_esm_afdb", [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)], ["below 0.5", "0.5 to 0.7", "0.7 to 0.9", "0.9 or more"],
       "TM-score between the ESMFold and AlphaFold models")
a1.set_ylabel("TED domains recovered (overlap ≥ 0.8, %)")
a1.set_title("By structural agreement with AlphaFold")
binned(a2, "esm_plddt", [(0, 50), (50, 70), (70, 85), (85, 101)], ["below 50", "50 to 70", "70 to 85", "85 or more"],
       "ESMFold mean pLDDT of the protein")
a2.set_title("By ESMFold confidence")
head(fig, "Sequence-only chopping matches TED when ESMFold reproduces the AlphaFold fold",
     "Sequence-only mode; n = published TED domains in each bin", y=0.975)
save(fig, "fig4_esmfold_quality")

# ------------------------------------------------------------------------------------------------------
# 5. Recovery by TED domain type
# ------------------------------------------------------------------------------------------------------
nted = {r["chain"]: int(r["n_ted"]) for r in pr}
groups = [
    ("TED consensus level", [("high\n(3 parsers)", lambda r: r["consensus_level"] == "high"),
                             ("medium\n(2 parsers)", lambda r: r["consensus_level"] == "medium")]),
    ("Domain shape", [("continuous", lambda r: r["num_segments"] == "1"),
                      ("discontinuous", lambda r: r["num_segments"] != "1")]),
    ("Domains in the protein", [("1", lambda r: nted[r["chain"]] == 1), ("2", lambda r: nted[r["chain"]] == 2),
                                ("3 or more", lambda r: nted[r["chain"]] >= 3)]),
]
fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.5), sharey=True, gridspec_kw={"width_ratios": [2, 2, 3]})
fig.subplots_adjust(left=0.07, right=0.985, top=0.72, bottom=0.2, wspace=0.08)
for ax, (title, gs) in zip(axes, groups):
    w = 0.24
    for k, (m, (name, col)) in enumerate(MODE.items()):
        t = [r for r in td if r["mode"] == m]
        ys, ns = [], []
        for _, f in gs:
            sel = [r for r in t if f(r)]
            ys.append(100 * sum(float(r["best_iou"]) >= 0.8 for r in sel) / max(1, len(sel)))
            ns.append(len(sel))
        xs = [i + (k - 0.5) * (w + 0.03) for i in range(len(gs))]
        ax.bar(xs, ys, width=w, color=col, label=name)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1.5, f"{y:.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(range(len(gs)), [f"{g[0]}\nn = {n}" for g, n in zip(gs, ns)])
    ax.set_title(title)
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.6, len(gs) - 0.4)
    ygrid(ax)
axes[0].set_ylabel("TED domains recovered (overlap ≥ 0.8, %)")
axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.30), ncol=2, columnspacing=1.6)
head(fig, "Which TED domains are harder to reproduce",
     "Share of published TED domains matched at overlap ≥ 0.8; n = TED domains in each group", y=0.975)
save(fig, "fig5_recovery_by_type")

# ------------------------------------------------------------------------------------------------------
# 6. CATH label: each tier against TED's label, on TED's own published domains
# ------------------------------------------------------------------------------------------------------
tier_names = [("exact lookup", "Exact lookup\n(MD5 in TED)"), ("structure", "Structure\n(Foldseek, then Foldclass)"),
              ("sequence", "Sequence\n(TED cluster transfer)"),
              ("sequence (own cluster removed)", "Sequence, domain's own\ncluster removed"),
              ("nearest labelled homolog (own cluster removed)", "Nearest labelled homolog,\nown cluster removed")]
OUT = [("same label as TED", DBLUE, lambda o: o in ("same superfamily", "same fold")),
       ("same fold, different or no superfamily", MBLUE, lambda o: o == "same fold only"),
       ("different label", ORANGE, lambda o: o == "different"), ("no label", GRAY, lambda o: o == "no label")]


def stacked(ax, rowsets, names, height=0.42):
    for i, (rs, nm) in enumerate(zip(rowsets, names)):
        left = 0
        for label, col, f in OUT:
            v = 100 * sum(f(r["outcome"]) for r in rs) / max(1, len(rs))
            ax.barh(i, v, left=left, height=height, color=col, edgecolor=SURFACE, linewidth=2,
                    label=label if i == 0 else None)
            if v >= 6:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=9.5,
                        color="white" if col in (DBLUE, MBLUE, ORANGE) else INK)
            left += v
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)


lab = [r for r in tiers if r["ted_level"] != "-"]
unl = [r for r in tiers if r["ted_level"] == "-"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.9), gridspec_kw={"width_ratios": [3.1, 1.25]})
fig.subplots_adjust(left=0.19, right=0.975, top=0.70, bottom=0.11, wspace=0.13)
stacked(a1, [[r for r in lab if r["tier"] == t] for t, _ in tier_names], [n for _, n in tier_names])
a1.set_title(f"Domains TED labelled (n = {len(lab)//len(tier_names)})")
fig.legend(*a1.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.005, 0.865), ncol=4, columnspacing=1.4, handlelength=1.1)
vals = [100 * sum(r["outcome"].endswith("ours labelled") for r in unl if r["tier"] == t) / max(1, sum(r["tier"] == t for r in unl))
        for t, _ in tier_names]
a2.barh(range(len(tier_names)), vals, height=0.42, color=AQUA)
for i, v in enumerate(vals):
    a2.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=9.5, color=INK)
a2.set_yticks(range(len(tier_names)), [""] * len(tier_names))
a2.invert_yaxis()
a2.set_xlim(0, 100)
a2.set_xticks([0, 50, 100], ["0", "50", "100%"])
a2.spines["left"].set_visible(False)
a2.xaxis.grid(True)
a2.set_axisbelow(True)
a2.set_title(f"TED-unlabelled (n = {len(unl)//len(tier_names)}):\nshare we label")
head(fig, "Each labelling route against TED's own label",
     "TED's published domains, cut from the AlphaFold models. 'Same label' = same superfamily where TED gave one, same fold where TED gave a fold",
     y=0.975)
save(fig, "fig6_label_tiers")

# ------------------------------------------------------------------------------------------------------
# 7. End to end: published TED-labelled domains -> recovered with TED's label?
# ------------------------------------------------------------------------------------------------------
by = {m: {r["matched_ted_id"]: r for r in e2e if r["mode"] == m and r["matched_ted_id"] and float(r["iou"]) >= 0.8} for m in MODE}
E2E = [("domain recovered, same label as TED", DBLUE), ("recovered, same fold only", MBLUE), ("recovered, different label", ORANGE),
       ("recovered, no label", GRAY), ("domain not recovered (overlap < 0.8)", LGRAY)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})
fig.subplots_adjust(left=0.15, right=0.975, top=0.69, bottom=0.24, wspace=0.42)
ROWS7 = [("afdb", "outcome", "AlphaFold model\n(TED's input)"), ("afdb", "novel_outcome", "AlphaFold model,\nas if new to TED*"),
         ("esm", "outcome", "Sequence only\n(ESMFold model)"), ("esm", "novel_outcome", "Sequence only,\nas if new to TED*")]
for i, (m, okey, _) in enumerate(ROWS7):
    t = [r for r in td if r["mode"] == m and r["ted_cath_level"] != "-"]
    c = Counter()
    for r in t:
        e = by[m].get(r["ted_id"])
        if e is None:
            c[4] += 1
        else:
            o = e[okey]
            c[0 if o in ("same superfamily", "same fold") else 1 if o == "same fold only" else 2 if o == "different" else 3] += 1
    left = 0
    for k, (label, col) in enumerate(E2E):
        v = 100 * c[k] / len(t)
        a1.barh(i, v, left=left, height=0.46, color=col, edgecolor=SURFACE, linewidth=2, label=label if i == 0 else None)
        if v >= 5:
            a1.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=9.5,
                    color="white" if col in (DBLUE, MBLUE, ORANGE) else INK)
        left += v
    n_lab = len(t)
a1.set_yticks(range(4), [r[2] for r in ROWS7])
a1.set_ylim(3.55, -0.55)
a1.set_xlim(0, 100)
a1.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
a1.spines["left"].set_visible(False)
a1.xaxis.grid(True)
a1.set_axisbelow(True)
a1.set_title(f"TED-labelled domains (n = {n_lab}): chopped and labelled like TED?")
fig.legend(*a1.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.005, 0.865), ncol=3, columnspacing=1.4, handlelength=1.1, fontsize=9)
srcs = [("ted-exact", "Exact lookup"), ("foldseek", "Foldseek"), ("foldclass", "Foldclass"), ("mmseqs-transfer", "Sequence cluster"),
        ("none", "No label")]
hh = 0.26
for k, (m, (name, col)) in enumerate(MODE.items()):
    e = [r for r in e2e if r["mode"] == m]
    vals = [100 * sum(r["source"] == s for r in e) / len(e) for s, _ in srcs]
    ys = [i + (k - 0.5) * (hh + 0.04) for i in range(len(srcs))]
    a2.barh(ys, vals, height=hh, color=col, label=name)
    for y, v in zip(ys, vals):
        if v >= 0.5:
            a2.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=9, color=INK2)
a2.set_yticks(range(len(srcs)), [n for _, n in srcs])
a2.invert_yaxis()
a2.set_xlim(0, 100)
a2.set_xticks([0, 50, 100], ["0", "50", "100%"])
a2.spines["left"].set_visible(False)
a2.xaxis.grid(True)
a2.set_axisbelow(True)
a2.set_title("Which route decided the label\n(all our domains)")
a2.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.09), ncol=1, fontsize=9)
fig.text(0.012, 0.035, "* exact lookup switched off and the domain's own TED sequence cluster removed from the MMseqs2 database: "
         "what a protein that is not in TED would get", fontsize=8.6, color=INK2, ha="left", va="bottom")
head(fig, "End to end: do we give TED's domains TED's labels?",
     "Chopping and labelling combined. A domain counts as recovered when one of ours overlaps it at ≥ 0.8", y=0.975)
save(fig, "fig7_end_to_end_labels")

# ------------------------------------------------------------------------------------------------------
# 8. Example proteins: TED vs ours (AlphaFold model) vs ours (sequence only)
# ------------------------------------------------------------------------------------------------------
prot = json.load(open(here / "proteins.json"))
chops = {m: {(r["name"][:-4] if m == "esm" else r["name"]): r for r in json.load(open(here / f"chop_{m}.json"))["results"]}
         for m in MODE}
pa = {r["chain"]: r for r in pr if r["mode"] == "afdb"}
pe = {r["chain"]: r for r in pr if r["mode"] == "esm"}
both = [c for c in prot if c in pa and c in pe]


def pick(cond, key, n, taken):
    c = sorted((c for c in both if c not in taken and cond(c)), key=key)[:n]
    taken.update(c)
    return c


taken = set()
ex = []
ex += [(c, "both modes identical to TED") for c in pick(lambda c: int(pa[c]["n_ted"]) >= 3 and pa[c]["all_exact"] == "1" and pe[c]["all_exact"] == "1",
                                                          lambda c: -int(pa[c]["n_ted"]), 2, taken)]
ex += [(c, "discontinuous domain") for c in pick(lambda c: any(d["num_segments"] > 1 for d in prot[c]["ted"]) and float(pe[c]["mean_best_iou"]) > 0.85,
                                                   lambda c: -int(pa[c]["nres"]), 1, taken)]
ex += [(c, "sequence mode: boundaries shifted") for c in pick(lambda c: pe[c]["n_ours"] == pe[c]["n_ted"] and 0.6 < float(pe[c]["mean_best_iou"]) < 0.9 and int(pe[c]["n_ted"]) >= 2,
                                                                lambda c: float(pe[c]["mean_best_iou"]), 1, taken)]
ex += [(c, "sequence mode: extra domains") for c in pick(lambda c: int(pe[c]["n_ours"]) > int(pe[c]["n_ted"]) and pa[c]["all_exact"] == "1",
                                                           lambda c: -int(pa[c]["nres"]), 1, taken)]
ex += [(c, "sequence mode: domains missed") for c in pick(lambda c: int(pe[c]["n_ours"]) < int(pe[c]["n_ted"]) and pa[c]["all_exact"] == "1",
                                                            lambda c: -int(pa[c]["n_ted"]), 1, taken)]
ex += [(c, "ESMFold fold differs (low TM-score)") for c in pick(lambda c: pe[c]["tm_esm_afdb"] not in ("", "None") and float(pe[c]["tm_esm_afdb"]) < 0.45 and int(pe[c]["n_ted"]) >= 1,
                                                                  lambda c: float(pe[c]["tm_esm_afdb"]), 1, taken)]
ex += [(c, "AlphaFold-model mode differs from TED") for c in pick(lambda c: pa[c]["all_exact"] == "0", lambda c: float(pa[c]["mean_best_iou"]), 1, taken)]

fig, axes = plt.subplots(len(ex), 1, figsize=(10.4, 1.16 * len(ex) + 1.5))
fig.subplots_adjust(left=0.215, right=0.985, top=1 - 1.25 / fig.get_figheight(), bottom=0.5 / fig.get_figheight(), hspace=0.62)
for ax, (c, why) in zip(axes, ex):
    L = prot[c]["nres"]
    ted = [parse_chopping(d["chopping"])[0] for d in prot[c]["ted"]]
    tsets = [{r for a, b in s for r in range(a, b + 1)} for s in ted]
    tracks = [("TED (published)", ted, list(range(len(ted))))]
    for m, nm in (("afdb", "Ours, AlphaFold model"), ("esm", "Ours, sequence only")):
        ours = [parse_chopping(d["chopping"])[0] for d in chops[m][c]["domains"]]
        cols = []
        for s in ours:
            os_ = {r for a, b in s for r in range(a, b + 1)}
            best = max(range(len(tsets)), key=lambda i: len(os_ & tsets[i]) / len(os_ | tsets[i]), default=None)
            cols.append(best if best is not None and len(os_ & tsets[best]) / len(os_ | tsets[best]) >= 0.5 else None)
        tracks.append((nm, ours, cols))
    for y, (nm, doms, cols) in enumerate(tracks):
        ax.plot([1, L], [y, y], color=AXIS, lw=1, zorder=1)
        for segs, ci in zip(doms, cols):
            col = CAT8[ci % 8] if ci is not None else MUTED
            for a, b in segs:
                ax.barh(y, b - a + 1, left=a - 0.5, height=0.58, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if len(segs) > 1:
                ax.plot([segs[0][1], segs[-1][0]], [y, y], color=col, lw=1.6, zorder=2)
    ax.set_yticks(range(3), [t[0] for t in tracks], fontsize=8.8)
    ax.set_ylim(2.6, -0.6)
    ax.set_xlim(1, L)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    acc = c.split("-")[1]
    ax.set_title(f"{acc}  ·  {L} residues  ·  {why}", fontsize=9.5, fontweight="bold", loc="left", pad=3)
axes[-1].set_xlabel("Residue number")
head(fig, "Example proteins: TED's domains and ours", "Each box is a domain (a line joins the pieces of a discontinuous one). "
     "Our domains take the colour of the TED domain\nthey overlap (≥ 0.5); gray = no TED counterpart", y=1 - 0.12 / fig.get_figheight())
save(fig, "fig8_examples")
json.dump([{"chain": c, "why": w} for c, w in ex], open(res / "example_proteins.json", "w"), indent=1)
