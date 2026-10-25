# 11_shift_validation.py - re-checks Sections 5.4 and 5.5 with matched sample sizes.
# (a) Centroid shift: 40 test set 1 images per class against the 10 test set 2 images (observed) or the other
#     10 test set 1 images (reference), over repeated draws; the exceedance rate is used descriptively.
# (b) Variance decomposition: within-class and between-class variance on test set 2 and on test set 1 at 10 per class.
# Trains nothing; run after 05_generalisation_analysis.py.

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PROC, OUT = ROOT / "data" / "processed", ROOT / "data" / "output"
SEED, DRAWS, REPEATS, N_CLASSES = 7, 200, 5, 1000
rng = np.random.default_rng(SEED)

X = {t: np.load(PROC / f"{t}_X.npy").astype(np.float64) for t in ("test1", "test2")}
y = {t: np.load(PROC / f"{t}_y.npy").astype(int) for t in X}
gap = pd.read_csv(OUT / "gen_perclass_A.csv").set_index("class").gap


# (a) Matched-size centroid shift
rows = []
for k in range(N_CLASSES):
    A, B = X["test1"][y["test1"] == k], X["test2"][y["test2"] == k]
    cB = B.mean(axis=0)
    obs, null = np.empty(DRAWS), np.empty(DRAWS)
    for d in range(DRAWS):
        s = rng.permutation(len(A))
        c40 = A[s[:40]].mean(axis=0)
        obs[d] = np.linalg.norm(c40 - cB)
        null[d] = np.linalg.norm(c40 - A[s[40:50]].mean(axis=0))
    rows.append({"class": k, "observed": obs.mean(), "null_mean": null.mean(), "null_95": np.percentile(null, 95),
                 "excess": obs.mean() - null.mean(), "p_value": ((null >= obs).sum() + 1) / (DRAWS + 1)})
shift = pd.DataFrame(rows)
shift["q_value"] = stats.false_discovery_control(shift.p_value)
shift["gap_A"] = shift["class"].map(gap)
shift.to_csv(OUT / "shift_validation_classes.csv", index=False)

summary = {
    "mean_observed_distance": shift.observed.mean(),
    "mean_null_distance": shift.null_mean.mean(),
    "share_observed_above_null_mean": (shift.observed > shift.null_mean).mean(),
    "share_observed_above_null_95th": (shift.observed > shift.null_95).mean(),
    "share_significant_bh_5pct": (shift.q_value < 0.05).mean(),
    "pearson_excess_vs_gap_A": stats.pearsonr(shift.excess, shift.gap_A)[0],
    "spearman_excess_vs_gap_A": stats.spearmanr(shift.excess, shift.gap_A)[0],
}


# (b) Within- and between-class variance, summed over features
def decompose(Z, labels):
    grand = Z.mean(axis=0)
    means = np.array([Z[labels == k].mean(axis=0) for k in range(N_CLASSES)])
    counts = np.bincount(labels, minlength=N_CLASSES)[:, None]
    between = (counts * (means - grand) ** 2).sum(axis=0) / len(Z)
    within = ((Z - means[labels]) ** 2).sum(axis=0) / len(Z)
    return within.sum(), between.sum()


w1, b1 = np.mean([decompose(X["test1"][r], y["test1"][r]) for r in
                  (np.concatenate([rng.choice(np.flatnonzero(y["test1"] == k), 10, replace=False)
                                   for k in range(N_CLASSES)]) for _ in range(REPEATS))], axis=0)
w2, b2 = decompose(X["test2"], y["test2"])
variance = pd.DataFrame({"test_set": ["test1 (10 per class)", "test2"], "within_class": [w1, w2],
                         "between_class": [b1, b2]})
variance["between_share"] = variance.between_class / (variance.within_class + variance.between_class)
variance.to_csv(OUT / "variance_decomposition.csv", index=False)
summary.update({"within_class_change_pct": 100 * (w2 / w1 - 1), "between_class_change_pct": 100 * (b2 / b1 - 1)})
pd.Series(summary).to_csv(OUT / "shift_validation_summary.csv", header=["value"])
print(pd.Series(summary).round(4).to_string())
print(variance.round(4).to_string(index=False))


# Figure: observed against matched null distance per class; within- and between-class variance
BLUE, ORANGE = "#2a78d6", "#eb6834"
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sig = shift.q_value < 0.05
axes[0].scatter(shift.null_mean[~sig], shift.observed[~sig], s=8, color=BLUE, alpha=0.5, label="not significant")
axes[0].scatter(shift.null_mean[sig], shift.observed[sig], s=8, color=ORANGE, alpha=0.7, label="significant (FDR 5%)")
lim = [min(shift.null_mean.min(), shift.observed.min()), max(shift.null_mean.max(), shift.observed.max())]
axes[0].plot(lim, lim, "k--", lw=1)
axes[0].set_xlabel("null distance (test set 1, 40 vs 10)")
axes[0].set_ylabel("observed distance (test set 1 40 vs test set 2 10)")
axes[0].set_title("Matched-size centroid shift per class")
axes[0].legend(frameon=False, fontsize=8)
x = np.arange(2)
axes[1].bar(x - 0.2, variance.within_class, 0.4, color=BLUE, label="within-class")
axes[1].bar(x + 0.2, variance.between_class, 0.4, color=ORANGE, label="between-class")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["test set 1\n(10 per class)", "test set 2"])
axes[1].set_ylabel("variance, summed over features")
axes[1].set_title("Variance decomposition")
axes[1].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig_shift_validation.png", dpi=150, bbox_inches="tight")
plt.close()
