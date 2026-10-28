# 10_report_figures.py - extra report figures and summary statistics, computed from the outputs of 01 to 09
# Reads data/output/*.csv and data/processed/*.npy and trains no classifier. Run after 09_results.py.
# Writes figx_*.png, report_stats.csv, report_group_gaps.csv, report_test1_duplicate_pairs.csv
# and report_refine_deciles.csv to data/output/.

import hashlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from common import ROOT, OUT, SEED, N_CLASSES, load, typicality, typicality_label_free, read_predictions

DST = OUT
BLUE, ORANGE, TEAL, GREY, INK = "#2a78d6", "#eb6834", "#1a9e77", "#8a8a8a", "#333333"
CLS = {"Regularised linear (A)": BLUE, "Small MLP (B)": ORANGE, "Nearest centroid (C)": TEAL}
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.edgecolor": "#999999", "figure.dpi": 100})
rng = np.random.default_rng(SEED)
STATS = []


def stat(name, value, source):
    STATS.append({"statistic": name, "value": value, "source": source})


def tidy(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(DST / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. Paired differences from classifier A (forest plot), both bootstrap schemes
r = pd.read_csv(OUT / "results_paired.csv")
r["model"] = r.comparison.str.split(" - ").str[0]
names = {"Small MLP (B)": "B minus A", "Nearest centroid (C)": "C minus A", "Refined linear (A + R1/R2)": "R1 minus A"}
qty = {"top-1 difference, test1": "Test set 1 top-1", "top-1 difference, test2": "Test set 2 top-1", "gap difference": "Gap"}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
for ax, (q, qlabel) in zip(axes, qty.items()):
    for i, (m, mlabel) in enumerate(names.items()):
        for off, scheme, color in ((-0.12, "image", BLUE), (0.12, "class", ORANGE)):
            row = r[(r.model == m) & (r.quantity == q) & (r.bootstrap == scheme)].iloc[0]
            ax.plot([row.ci_low, row.ci_high], [i + off] * 2, color=color, lw=2.2, solid_capstyle="round")
            ax.plot(row.estimate, i + off, "o", ms=7, color=color, mec="white", mew=1.5)
    ax.axvline(0, color=GREY, lw=1, ls="--")
    ax.set_title(qlabel)
    ax.set_xlabel("difference from A (points)")
    tidy(ax, "x")
axes[0].set_yticks(range(len(names)), list(names.values()))
axes[0].invert_yaxis()
fig.legend([Line2D([], [], color=BLUE, lw=2.2, marker="o"), Line2D([], [], color=ORANGE, lw=2.2, marker="o")],
           ["resampling images", "resampling classes"], frameon=False, loc="lower center", ncol=2, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.08))
save(fig, "figx_paired.png")

# 2. MLP tuning grid as heatmaps, one panel per dropout rate
m = pd.read_csv(OUT / "mlp_tuning.csv")
lo, hi = m.val_top1.min() * 100, m.val_top1.max() * 100
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, d in zip(axes, sorted(m.dropout.unique())):
    part = m[m.dropout == d].pivot(index="hidden", columns="weight_decay", values="val_top1") * 100
    im = ax.imshow(part.values, cmap="Blues", vmin=lo - 0.05, vmax=hi, aspect="auto", origin="lower")
    for (i, j), v in np.ndenumerate(part.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5, color="white" if v > (lo + hi) / 2 + 0.05 else INK)
    ax.set_xticks(range(3), [f"{w:g}" for w in part.columns])
    ax.set_yticks(range(3), [f"{h:,}" for h in part.index] if d == 0 else [""] * 3)
    ax.set_xlabel("weight decay")
    ax.set_title(f"dropout {d:g}")
axes[0].set_ylabel("hidden units")
fig.colorbar(im, ax=axes, shrink=0.85, label="validation top-1 (%)")
fig.savefig(DST / "figx_mlp_grid.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 3. Gap with and without the 798 duplicates
o = pd.read_csv(OUT / "gen_overall.csv")
fig, ax = plt.subplots(figsize=(6.6, 2.8))
for i, c in enumerate(CLS):
    for off, sub, mk in ((-0.12, "all", "o"), (0.12, "test1 without duplicates", "s")):
        row = o[(o.classifier == c) & (o.subset == sub)].iloc[0]
        ax.plot([100 * row.gap_ci_low, 100 * row.gap_ci_high], [i + off] * 2, color=CLS[c], lw=2.2)
        ax.plot(100 * row.gap, i + off, mk, ms=7, color=CLS[c], mec="white", mew=1.5)
        ax.text(100 * row.gap_ci_high + 0.08, i + off, f"{100 * row.gap:.2f}", va="center", fontsize=9, color=INK)
ax.set_yticks(range(3), list(CLS))
ax.invert_yaxis()
ax.set_xlabel("top-1 gap, test set 1 minus test set 2 (points), 95% CI")
ax.legend([Line2D([], [], color=INK, marker="o", ls=""), Line2D([], [], color=INK, marker="s", ls="")],
          ["all of test set 1", "test set 1 without the 798 duplicates"], frameon=False, fontsize=9, loc="lower center",
          ncol=2, bbox_to_anchor=(0.5, 1.0))
ax.set_xlim(6.3, 9.7)
tidy(ax, "x")
save(fig, "figx_duplicates.png")

# 4. Per-class view: accuracy on both test sets, and agreement between classifiers
pa, pc = pd.read_csv(OUT / "gen_perclass_linear.csv"), pd.read_csv(OUT / "gen_perclass_centroid.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
j1, j2 = rng.uniform(-0.008, 0.008, 1000), rng.uniform(-0.02, 0.02, 1000)
sig = pa.q_value < 0.05
axes[0].scatter(pa.acc_test1[~sig] + j1[~sig], pa.acc_test2[~sig] + j2[~sig], s=9, alpha=0.45, color=BLUE, lw=0, label="not significant")
axes[0].scatter(pa.acc_test1[sig] + j1[sig], pa.acc_test2[sig] + j2[sig], s=22, color=ORANGE, lw=0, label="significant (FDR 5%)")
axes[0].plot([0, 1], [0, 1], "--", color=GREY, lw=1)
axes[0].set_xlabel("class top-1 on test set 1 (50 images)")
axes[0].set_ylabel("class top-1 on test set 2 (10 images)")
axes[0].set_title("Classifier A, one point per class")
axes[0].legend(frameon=False, loc="upper left", fontsize=9)
axes[1].scatter(100 * pa.gap + rng.uniform(-1, 1, 1000), 100 * pc.gap + rng.uniform(-1, 1, 1000), s=9, alpha=0.45, color=TEAL, lw=0)
axes[1].plot([-80, 90], [-80, 90], "--", color=GREY, lw=1)
axes[1].set_xlabel("class gap, classifier A (points)")
axes[1].set_ylabel("class gap, classifier C (points)")
axes[1].set_title(f"Same classes drop for both (r = {np.corrcoef(pa.gap, pc.gap)[0, 1]:.2f})")
for ax in axes:
    tidy(ax, None)
save(fig, "figx_classes.png")

# 5. Gap by broad class group (ImageNet index ranges)
def group(i):
    return ("Dog breeds" if 151 <= i <= 268 else "Other animals" if i <= 397 else "Objects" if i <= 923
            else "Food, plants and scenes")
pa["group"] = pa["class"].map(group)
order = ["Other animals", "Dog breeds", "Objects", "Food, plants and scenes"]
rows = []
for g in order:
    d = pa[pa.group == g]
    boot = [d.gap.values[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    rows.append({"group": g, "n": len(d), "gap": 100 * d.gap.mean(), "lo": 100 * np.percentile(boot, 2.5),
                 "hi": 100 * np.percentile(boot, 97.5), "share": d.gap.sum() / pa.gap.sum()})
G = pd.DataFrame(rows)
G.to_csv(OUT / "report_group_gaps.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
y = np.arange(len(G))
axes[0].barh(y, G.gap, height=0.55, color=BLUE)
axes[0].errorbar(G.gap, y, xerr=[G.gap - G.lo, G.hi - G.gap], fmt="none", ecolor=INK, capsize=3, lw=1.2)
for i, r_ in G.iterrows():
    axes[0].text(r_.hi + 0.3, i, f"{r_.gap:.1f}", va="center", fontsize=9.5)
axes[0].axvline(100 * pa.gap.mean(), color=ORANGE, ls="--", lw=1.2)
axes[0].text(100 * pa.gap.mean() + 0.2, -0.55, "all classes", color=ORANGE, fontsize=9)
axes[0].set_yticks(y, [f"{g} ({n})" for g, n in zip(G.group, G.n)])
axes[0].invert_yaxis()
axes[0].set_xlabel("mean class gap, classifier A (points), 95% CI")
axes[1].barh(y, 100 * G.share, height=0.55, color=TEAL)
for i, r_ in G.iterrows():
    axes[1].text(100 * r_.share + 1, i, f"{100 * r_.share:.0f}%", va="center", fontsize=9.5)
axes[1].set_yticks(y, [""] * len(G))
axes[1].invert_yaxis()
axes[1].set_xlim(0, 80)
axes[1].set_xlabel("share of the total gap (%)")
for ax in axes:
    tidy(ax, "x")
save(fig, "figx_groups.png")

# 6. How the errors of the ten strongest drops split
e = pd.read_csv(OUT / "gen_error_patterns.csv")
cls = list(dict.fromkeys(e.true_name))
fig, ax = plt.subplots(figsize=(8.6, 3.8))
for i, c in enumerate(cls):
    d = e[e.true_name == c]
    total = round(d["count"].iloc[0] / d.share_of_errors.iloc[0])
    top = d.iloc[0]
    ax.barh(i, top["count"], color=ORANGE, height=0.6)
    ax.barh(i, total - top["count"], left=top["count"], color="#c9d6e8", height=0.6)
    ax.text(total + 0.15, i, f"{top.predicted_name} ({top['count']} of {total})", va="center", fontsize=9)
ax.set_yticks(range(len(cls)), cls)
ax.invert_yaxis()
ax.set_xlim(0, 15)
ax.set_xlabel("wrong predictions on test set 2 (of 10 images per class)")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color="#c9d6e8")],
          ["most common wrong class", "all other wrong classes"], frameon=False, loc="lower right", fontsize=9)
tidy(ax, "x")
save(fig, "figx_errors.png")

# 7. Centroid shift: observed distance against the same-size reference, per class
s = pd.read_csv(OUT / "gen_centroid_shift.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
bins = np.linspace(min(s.reference_mean.min(), s.observed.min()), s.observed.quantile(0.995), 45)
axes[0].hist(s.reference_mean, bins=bins, color=BLUE, alpha=0.75, label="reference: other 10 test set 1 images")
axes[0].hist(s.observed, bins=bins, color=ORANGE, alpha=0.65, label="test set 2's 10 images")
axes[0].set_xlabel("distance to the centroid of 40 test set 1 images")
axes[0].set_ylabel("number of classes")
axes[0].legend(frameon=False, fontsize=9)
axes[1].hist(s.excess, bins=45, color=TEAL)
axes[1].axvline(0, color=GREY, ls="--", lw=1)
axes[1].set_xlabel("excess shift of test set 2 (observed minus reference)")
axes[1].set_ylabel("number of classes")
axes[1].set_title(f"{(s.excess > 0).mean():.1%} of classes shift beyond the reference")
for ax in axes:
    tidy(ax)
save(fig, "figx_shift.png")

# 8. Per feature: size of the mean shift against the loss of separability
f = pd.read_csv(OUT / "gen_features.csv")
fig, ax = plt.subplots(figsize=(6.6, 3.8))
sc = ax.scatter(f.cohens_d.abs(), f.eta2_drop, c=f.eta2_train, cmap="viridis", s=10, alpha=0.8)
ax.set_xlabel("absolute mean shift per feature (|Cohen's d|)")
ax.set_ylabel("loss of class separability (η², test set 1 minus 2)")
fig.colorbar(sc, ax=ax, label="separability on train_small")
tidy(ax, None)
save(fig, "figx_feature_drop.png")

# 9. Typicality mix: share of each test set in each test set 1 decile
b = pd.read_csv(OUT / "gen_typicality_bins.csv")
b = b[(b.bins == 10) & (b.classifier == "Regularised linear (A)")]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
for ax, meas in zip(axes, ["own label", "label-free"]):
    d = b[b.measure == meas]
    ax.bar(d.bin - 0.2, 100 * d.share_test1, 0.38, color=BLUE, label="test set 1")
    ax.bar(d.bin + 0.2, 100 * d.share_test2, 0.38, color=ORANGE, label="test set 2")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel(f"{meas} typicality decile of test set 1 (1 = least typical)")
    ax.set_title(f"{meas}: {100 * d.share_test2.iloc[0]:.1f}% of test set 2 in decile 1")
    tidy(ax)
axes[0].set_ylabel("share of the test set (%)")
axes[0].legend(frameon=False)
save(fig, "figx_typ_mix.png")

# 10. Share of the gap reproduced by the typicality mix, all measures, bin counts and classifiers
t = pd.read_csv(OUT / "gen_typicality.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, meas in zip(axes, ["own label", "label-free"]):
    for k, (c, color) in enumerate(CLS.items()):
        d = t[(t.measure == meas) & (t.classifier == c)]
        ax.plot(np.arange(3) + (k - 1) * 0.12, 100 * d.share_of_gap_reproduced, "o", ms=8, color=color, mec="white", mew=1.5, label=c)
    ax.axhline(100, color=GREY, ls="--", lw=1)
    ax.set_xticks(range(3), ["5 bins", "10 bins", "20 bins"])
    ax.set_title(meas)
    tidy(ax)
axes[0].set_ylabel("share of the gap reproduced (%)")
axes[1].legend(frameon=False, fontsize=9, loc="upper left")
save(fig, "figx_typ_share.png")

# 11. Cluster size against cluster drop
cl = pd.read_csv(OUT / "gen_clusters.csv")
sig_c = cl.q_value < 0.05
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.scatter(cl.n_test2[~sig_c], 100 * cl.gap[~sig_c], s=40, color=BLUE, alpha=0.75, label="not significant", lw=0)
ax.scatter(cl.n_test2[sig_c], 100 * cl.gap[sig_c], s=40, color=ORANGE, alpha=0.85, label="significant (FDR 5%)", lw=0)
ax.axhline(0, color=GREY, ls="--", lw=1)
for (_, r_), (dx, dy, ha) in zip(cl.sort_values("gap", ascending=False).head(3).iterrows(),
                                 ((-7, 0, "right"), (8, 7, "left"), (8, -7, "left"))):   # labels kept apart
    ax.annotate(r_.example_classes.split(",")[0], (r_.n_test2, 100 * r_.gap), xytext=(dx, dy), textcoords="offset points",
                fontsize=8.5, va="center", ha=ha)
ax.set_xscale("log")
ax.set_xticks([20, 50, 100, 200, 500], ["20", "50", "100", "200", "500"])
ax.minorticks_off()
ax.set_xlabel("test set 2 images in the cluster (log scale)")
ax.set_ylabel("cluster drop, test set 1 minus 2 (points)")
ax.legend(frameon=False, fontsize=9)
tidy(ax, None)
save(fig, "figx_cluster_size.png")

# 12. Refinement screening: gain on atypical validation images with CI
g = pd.read_csv(OUT / "refine_grid.csv")
g = g[g.setting.str.startswith("R")].reset_index(drop=True)
labels = [f"R1, β = {b_:g}" if s_.startswith("R1") else f"R2, γ = {c_:g}" for s_, b_, c_ in zip(g.setting, g.beta, g.gamma)]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for i, r_ in g.iterrows():
    color = BLUE if r_.setting.startswith("R1") else ORANGE
    ax.plot([100 * r_.like_ci_low, 100 * r_.like_ci_high], [i, i], color=color, lw=2.2)
    ax.plot(100 * r_.gain_like_for_like, i, "D" if r_.chosen else "o", ms=8, color=color, mec="white", mew=1.5)
ax.axvline(0, color=GREY, ls="--", lw=1)
ax.set_yticks(range(len(g)), [l + ("  (chosen)" if c else "") for l, c in zip(labels, g.chosen)])
ax.invert_yaxis()
ax.set_xlabel("gain on the least typical 10% of validation images (points), 95% CI")
tidy(ax, "x")
save(fig, "figx_refine_grid.png")

# 13. Refinement on the full split: the selection rule epoch by epoch
h = pd.read_csv(OUT / "refine_history.csv")
a_val = pd.read_csv(OUT / "linear_final.csv").val_top1.iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
axes[0].plot(h.epoch, 100 * h.val_top1, "o-", color=BLUE, ms=6)
axes[0].axhline(100 * (a_val - 0.002), color=ORANGE, ls="--", lw=1.2)
axes[0].text(1, 100 * (a_val - 0.002) + 0.03, "rule: within 0.2 points of A", color=ORANGE, fontsize=9)
axes[0].set_ylabel("validation top-1 (%)")
axes[0].set_title("Overall validation accuracy")
axes[1].plot(h.epoch, 100 * h.val_atypical_top1, "o-", color=TEAL, ms=6)
k = h[h.kept].iloc[0]
axes[1].plot(k.epoch, 100 * k.val_atypical_top1, "D", ms=10, mfc="none", mec=INK, mew=1.5)
first_ok = int(h.epoch[h.meets_rule].min())
axes[1].axvspan(first_ok - 0.5, h.epoch.max() + 0.5, color="#eef2f8", zorder=0)
axes[1].text(first_ok - 0.4, 100 * h.val_atypical_top1.min(), "epochs that meet the rule", fontsize=9, color=INK)
axes[1].set_ylabel("atypical validation top-1 (%)")
axes[1].set_title("Least typical 10% of validation images")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.set_xticks(range(1, 11))
    tidy(ax)
save(fig, "figx_refine_history.png")



# 14. Summary statistics quoted in the report that no earlier script writes
pa, pb, pc = (pd.read_csv(OUT / f"gen_perclass_{c}.csv") for c in ("linear", "mlp", "centroid"))
pa["group"] = pa["class"].map(group)
gap_pts = np.round(100 * pa.gap).astype(int)                      # gaps come in whole points
stat("classes losing 30 points or more (A)", int((gap_pts >= 30).sum()), "gen_perclass_linear.csv")
stat("classes gaining 30 points or more (A)", int((gap_pts <= -30).sum()), "gen_perclass_linear.csv")
stat("correlation of class gaps, A and B", np.corrcoef(pa.gap, pb.gap)[0, 1], "gen_perclass_*.csv")
stat("correlation of class gaps, A and C", np.corrcoef(pa.gap, pc.gap)[0, 1], "gen_perclass_*.csv")
sig = [set(p["class"][p.q_value < 0.05]) for p in (pa, pb, pc)]
stat("classes significant for all three classifiers", len(sig[0] & sig[1] & sig[2]), "gen_perclass_*.csv")
rho, p_rho = stats.spearmanr(pa.acc_test1, pa.gap)
stat("Spearman, test set 1 class accuracy vs gap", rho, "gen_perclass_linear.csv")
stat("  p-value", p_rho, "gen_perclass_linear.csv")
stat("mean gap, classes below 80% on test set 1 (points)", 100 * pa.gap[pa.acc_test1 < 0.8].mean(), "gen_perclass_linear.csv")
stat("mean gap, classes from 90% to below 96% on test set 1 (points)",
     100 * pa.gap[(pa.acc_test1 >= 0.9) & (pa.acc_test1 < 0.96)].mean(), "gen_perclass_linear.csv")
H, p_kw = stats.kruskal(*[pa.gap[pa.group == g] for g in order])
stat("Kruskal-Wallis H across class groups", H, "gen_perclass_linear.csv")
stat("  p-value", p_kw, "gen_perclass_linear.csv")
for g in order:
    d = pa[(pa.group == g) & (pa.q_value < 0.05)]
    stat(f"significant drops / rises, {g}", f"{int((d.gap > 0).sum())} / {int((d.gap < 0).sum())}", "gen_perclass_linear.csv")
rho_c, p_c = stats.spearmanr(cl.n_test2, cl.gap)
stat("Spearman, cluster size vs cluster drop", rho_c, "gen_clusters.csv")
stat("  p-value", p_c, "gen_clusters.csv")
stat("significant clusters with more than 90 test set 2 images", int(((cl.q_value < 0.05) & (cl.n_test2 > 90)).sum()), "gen_clusters.csv")

# 15. Centroid shift: how much of the excess distance a wider spread alone would produce
X1, y1 = load("test1")
X2, y2 = load("test2")
ratio = np.empty(N_CLASSES)
for k in range(N_CLASSES):
    t1 = X1[y1 == k].astype(np.float64).var(axis=0, ddof=1).sum()
    t2 = X2[y2 == k].astype(np.float64).var(axis=0, ddof=1).sum()
    # expected distance with no shift: 40-image mean of test set 1 vs 10-image mean of test set 2,
    # relative to the reference (two disjoint test set 1 samples of 40 and 10)
    ratio[k] = np.sqrt(t1 / 40 + t2 / 10) / np.sqrt(t1 / 40 + t1 / 10)
s = pd.read_csv(OUT / "gen_centroid_shift.csv")
null = s.reference_mean.to_numpy() * ratio
stat("centroid distance, observed", s.observed.mean(), "gen_centroid_shift.csv")
stat("centroid distance, reference", s.reference_mean.mean(), "gen_centroid_shift.csv")
stat("centroid distance expected with no shift but test set 2 spread", null.mean(), "test1_X.npy, test2_X.npy")
stat("excess due to the wider spread", (null - s.reference_mean).mean(), "test1_X.npy, test2_X.npy")
stat("excess left as a shift", (s.observed - null).mean(), "test1_X.npy, test2_X.npy")
stat("share of classes beyond the no-shift expectation", (s.observed > null).mean(), "test1_X.npy, test2_X.npy")

# 16. Duplicate feature vectors inside test set 1 and the classes they connect
names = pd.read_csv(OUT / "gen_perclass_linear.csv").name.to_numpy()
h1 = [hashlib.md5(r.tobytes()).hexdigest() for r in np.asarray(X1)]
groups = pd.DataFrame({"h": h1, "y": y1}).groupby("h").y.apply(list)
pairs = [sorted(v) for v in groups if len(v) > 1]
dp = pd.DataFrame([{"class_a": a, "name_a": names[a], "class_b": b, "name_b": names[b], "index_gap": b - a}
                   for a, b in pairs]).sort_values(["class_a", "class_b"])
dp.to_csv(OUT / "report_test1_duplicate_pairs.csv", index=False)
stat("duplicate pairs inside test set 1", len(dp), "test1_X.npy")
stat("  of which within 10 class indices", int((dp.index_gap <= 10).sum()), "test1_X.npy")

# 17. Refinement against classifier A by typicality decile, both measures: paired difference with 95% interval
ys = {"test1": y1, "test2": y2}
typ = {"own label": {"test1": typicality(X1, y1), "test2": typicality(X2, y2)},
       "label-free": {"test1": typicality_label_free(X1), "test2": typicality_label_free(X2)}}
cA, cR = (read_predictions(m, "test2")[:, 0] == y2 for m in ("linear", "refined"))
rows = []
for meas, t in typ.items():
    edges = np.quantile(t["test1"], np.linspace(0, 1, 11))[1:-1]
    b2 = np.digitize(t["test2"], edges)
    for b in range(10):
        m = b2 == b
        diff = cR[m].astype(float) - cA[m]
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]   # paired bootstrap
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"measure": meas, "decile": b + 1, "n_test2": int(m.sum()), "acc_A": cA[m].mean(),
                     "acc_refined": cR[m].mean(), "difference": diff.mean(), "ci_low": lo, "ci_high": hi})
dec = pd.DataFrame(rows)
dec.to_csv(OUT / "report_refine_deciles.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, meas in zip(axes, typ):
    d = dec[dec.measure == meas]
    ax.axhline(0, color=GREY, lw=1)
    ax.errorbar(d.decile, 100 * d.difference, yerr=[100 * (d.difference - d.ci_low), 100 * (d.ci_high - d.difference)],
                fmt="o", ms=5, capsize=3, color=ORANGE)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel(f"{meas} typicality decile of test set 1 (1 = least typical)")
    ax.set_title(meas)
    tidy(ax)
axes[0].set_ylabel("refined minus A, test set 2 top-1 (points)")
save(fig, "figx_refine_deciles.png")
stat("refined minus A, least typical decile, own label (points)", 100 * dec.difference.iloc[0], "report_refine_deciles.csv")
stat("refined minus A, least typical decile, label-free (points)", 100 * dec.difference.iloc[10], "report_refine_deciles.csv")
stat("deciles whose paired interval excludes 0", int(((dec.ci_low > 0) | (dec.ci_high < 0)).sum()), "report_refine_deciles.csv")

# 18. Statistics behind Table 4.4 and Section 5.4.2
mt = pd.read_csv(OUT / "mlp_tuning.csv")
for col in ("hidden", "dropout", "weight_decay"):
    for level, v in mt.groupby(col).val_top1.mean().items():
        stat(f"MLP mean validation top-1, {col} = {level:g}", v, "mlp_tuning.csv")
gf = pd.read_csv(OUT / "gen_features.csv")
stat("Spearman, separability drop vs separability on train_small", stats.spearmanr(gf.eta2_drop, gf.eta2_train)[0], "gen_features.csv")
stat("Spearman, separability drop vs |Cohen's d|", stats.spearmanr(gf.eta2_drop, gf.cohens_d.abs())[0], "gen_features.csv")
drop_rank = gf.eta2_drop.rank(pct=True)
for i in gf.cohens_d.abs().nlargest(2).index:
    stat(f"feature {int(gf.feature[i])} (|d| = {abs(gf.cohens_d[i]):.3f}): percentile of its separability drop",
         drop_rank[i], "gen_features.csv")

# 19. Print layout of the per-feature figure (Figure 5.8): two panels above, one below
var = pd.read_csv(OUT / "gen_variance.csv")
fig = plt.figure(figsize=(10, 7.6))
gs = fig.add_gridspec(2, 2)
ax0, ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
ax0.hist(gf.cohens_d, bins=50, color=BLUE, edgecolor="white")
ax0.axvline(0, color=GREY, lw=1)
ax0.set_xlabel("Cohen's d per feature (test set 2 minus test set 1)")
ax0.set_ylabel("number of features")
ax0.set_title(f"Mean shift: {(gf.q_value < 0.05).sum()} of {len(gf)} significant")
lims = [0, max(gf.eta2_test1.max(), gf.eta2_test2.max()) * 1.05]
sc = ax1.scatter(gf.eta2_test1, gf.eta2_test2, c=gf.eta2_train, s=8, cmap="viridis")
ax1.plot(lims, lims, "--", color=GREY)
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_xlabel("class separability on test set 1 (η²)")
ax1.set_ylabel("class separability on test set 2 (η²)")
ax1.set_title("Class separability per feature")
fig.colorbar(sc, ax=ax1, label="separability on train_small")
labels = ["test set 1\n(10 per class)", "test set 2"]
xs = np.arange(2)
ax2.bar(xs - 0.2, var.within_class, 0.4, color=BLUE, label="within-class")
ax2.bar(xs + 0.2, var.between_class, 0.4, color=ORANGE, label="between-class")
ax2.set_xticks(xs, labels)
ax2.set_xlim(-0.6, 2.4)
ax2.set_ylabel("variance, summed over features")
ax2.set_title("Variance decomposition")
ax2.legend(frameon=False, loc="center right")
for ax in (ax0, ax1, ax2):
    tidy(ax, grid=None)
save(fig, "figx_features_print.png")

# 20. Print layout of the typicality figure (Figure 5.11): two panels above, one below
tb = pd.read_csv(OUT / "gen_typicality_bins.csv")
ts = pd.read_csv(OUT / "gen_typicality.csv")
A_NAME = "Regularised linear (A)"
fig = plt.figure(figsize=(10, 7.6))
gs = fig.add_gridspec(2, 2)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
own = typ["own label"]
sim_bins = np.linspace(min(own["test1"].min(), own["test2"].min()), 1, 50)
for t, color in (("test1", BLUE), ("test2", ORANGE)):
    axes[0].hist(own[t], bins=sim_bins, density=True, histtype="step", lw=1.6, color=color,
                 label=f"test set {t[-1]} (mean {own[t].mean():.3f})")
axes[0].set_xlabel("cosine similarity to own-class training centroid")
axes[0].set_ylabel("density")
axes[0].set_title("Typicality (own label)")
axes[0].legend(frameon=False, loc="upper left")
for ax, meas in zip(axes[1:], ("own label", "label-free")):
    part = tb[(tb.measure == meas) & (tb.bins == 10) & (tb.classifier == A_NAME)]
    share = ts[(ts.measure == meas) & (ts.bins == 10) & (ts.classifier == A_NAME)].share_of_gap_reproduced.iloc[0]
    for t, color in (("test1", BLUE), ("test2", ORANGE)):
        acc = part[f"acc_{t}"]
        cnt = part.n_test2 if t == "test2" else len(y1) / 10
        ax.errorbar(part.bin, acc, yerr=1.96 * np.sqrt(acc * (1 - acc) / cnt), fmt="o-", ms=3, capsize=2,
                    color=color, label=f"test set {t[-1]}")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel(f"{meas} typicality decile of test set 1 (1 = least typical)")
    ax.set_ylabel("top-1, classifier A")
    ax.set_title(f"{meas}: reproduces {share:.0%} of the gap")
    ax.legend(frameon=False, loc="lower right")
for ax in axes:
    tidy(ax, grid=None)
save(fig, "figx_typicality_print.png")

# 21. Cluster accuracy on both test sets (Figure 5.13), with labels placed clear of the markers
fig, ax = plt.subplots(figsize=(7, 5))
sig_c = cl.q_value < 0.05
for mask, color, label in ((~sig_c, BLUE, "gap not significant"), (sig_c, ORANGE, "significant gap (FDR 5%)")):
    ax.scatter(100 * cl.acc_test1[mask], 100 * cl.acc_test2[mask], s=1.2 * cl.n_test2[mask], alpha=0.55,
               color=color, edgecolor="white", lw=0.5, label=label)
lo = 100 * min(cl.acc_test1.min(), cl.acc_test2.min()) - 3
ax.plot([lo, 100], [lo, 100], "--", color=GREY, lw=1)
top = cl.sort_values("gap", ascending=False).head(4).sort_values("acc_test2")
label_y = []                                   # labels in the empty space right of the points, at least 4 points apart
for v in 100 * top.acc_test2:
    label_y.append(max(v, label_y[-1] + 4) if label_y else v)
for (_, r), ly in zip(top.iterrows(), label_y):
    ax.annotate(r.example_classes.split(",")[0], (100 * r.acc_test1, 100 * r.acc_test2), xytext=(97, ly),
                textcoords="data", ha="left", va="center", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7, shrinkA=2, shrinkB=4))
ax.set_xlim(right=112)
ax.set_xticks(range(50, 101, 10))
ax.set_xlabel("top-1 on test set 1 (%)")
ax.set_ylabel("top-1 on test set 2 (%)")
ax.set_title("k-means clusters, classifier A; size = test set 2 images")
leg = ax.legend(frameon=False, loc="upper left")
for h in leg.legend_handles:
    h.set_sizes([30])
tidy(ax, grid=None)
save(fig, "figx_clusters.png")

pd.DataFrame(STATS).to_csv(OUT / "report_stats.csv", index=False)
print(pd.DataFrame(STATS).to_string(index=False))
