# 09_typicality_robustness.py - checks whether the typicality resultư depends on the number of bins.
# Typicality is computed as in 05_generalisation_analysis.py. For 5, 10 and 20 bins it reports per-bin accuracy,
# the standardised test set 2 prediction and its bootstrap residual. Trains nothing; run after the classifier scripts.
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PROC, OUT = ROOT / "data" / "processed", ROOT / "data" / "output"
SEED, DRAWS, BIN_COUNTS, N_CLASSES = 7, 1000, (5, 10, 20), 1000
rng = np.random.default_rng(SEED)


# 1. Typicality, as in Section 5.6
def unit(Z):
    Z = np.asarray(Z, dtype=np.float64)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


X_small = np.load(PROC / "train_small_X.npy")
y_small = np.load(PROC / "train_small_y.npy").astype(int)
centroid = np.zeros((N_CLASSES, X_small.shape[1]))
np.add.at(centroid, y_small, unit(X_small))
centroid = unit(centroid)
y = {t: np.load(PROC / f"{t}_y.npy").astype(int) for t in ("test1", "test2")}
typical = {t: (unit(np.load(PROC / f"{t}_X.npy")) * centroid[y[t]]).sum(axis=1) for t in y}


# 2. Top-1 correctness of each classifier, checked against the labels
def read_correct(t):
    a = pd.read_csv(OUT / f"classifier_a_{t}_predictions.csv", usecols=["true_label", "predicted_label"])
    b = pd.read_csv(OUT / "mlp_predictions.csv")
    b = b[b.test_set == t].sort_values("row")
    c = pd.read_csv(OUT / f"ncm_predictions_{t}.csv").sort_values("row")
    out = {}
    for name, lab, pred in (("A", a.true_label, a.predicted_label), ("B", b.label, b.pred), ("C", c.label, c.pred1)):
        if not np.array_equal(lab.to_numpy(), y[t]):
            raise ValueError(f"classifier {name}, {t}: labels do not match {t}_y.npy")
        out[name] = (lab.to_numpy() == pred.to_numpy()).astype(float)
    return out


correct = {t: read_correct(t) for t in y}


# 3. Standardisation for each number of bins
def predicted_test2(bins1, bins2, c1, n_bins):
    n1 = np.bincount(bins1, minlength=n_bins)
    acc1 = np.bincount(bins1, weights=c1, minlength=n_bins) / np.maximum(n1, 1)
    share2 = np.bincount(bins2, minlength=n_bins) / len(bins2)
    return (share2 * acc1)[n1 > 0].sum()


summary, detail = [], []
for n_bins in BIN_COUNTS:
    edges = np.quantile(typical["test1"], np.linspace(0, 1, n_bins + 1))[1:-1]
    bins = {t: np.digitize(typical[t], edges) for t in y}
    for clf in ("A", "B", "C"):
        c1, c2 = correct["test1"][clf], correct["test2"][clf]
        pred = predicted_test2(bins["test1"], bins["test2"], c1, n_bins)
        actual, acc1 = c2.mean(), c1.mean()
        boot = []
        for _ in range(DRAWS):
            i = rng.integers(0, len(c1), len(c1))
            j = rng.integers(0, len(c2), len(c2))
            boot.append(c2[j].mean() - predicted_test2(bins["test1"][i], bins["test2"][j], c1[i], n_bins))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        summary.append({"classifier": clf, "bins": n_bins, "test1_top1": acc1, "test2_top1": actual,
                        "test2_predicted": pred, "residual": actual - pred, "residual_ci_low": lo,
                        "residual_ci_high": hi, "share_of_gap_reproduced": (acc1 - pred) / (acc1 - actual)})
        for b in range(n_bins):
            m1, m2 = bins["test1"] == b, bins["test2"] == b
            p1, p2 = c1[m1].mean(), c2[m2].mean() if m2.any() else np.nan
            se = np.sqrt(p1 * (1 - p1) / m1.sum() + (p2 * (1 - p2) / m2.sum() if m2.any() else np.nan))
            detail.append({"classifier": clf, "bins": n_bins, "bin": b + 1, "n_test1": int(m1.sum()),
                           "n_test2": int(m2.sum()), "acc_test1": p1, "acc_test2": p2, "difference": p2 - p1,
                           "difference_ci_low": p2 - p1 - 1.96 * se, "difference_ci_high": p2 - p1 + 1.96 * se})

summary, detail = pd.DataFrame(summary), pd.DataFrame(detail)
summary.to_csv(OUT / "typicality_robustness_summary.csv", index=False)
detail.to_csv(OUT / "typicality_robustness_bins.csv", index=False)
print(summary.round(4).to_string(index=False))
d = detail[(detail.classifier == "A")]
print("\nclassifier A: bins whose 95% interval for (test2 - test1) excludes zero:")
print(d[(d.difference_ci_low > 0) | (d.difference_ci_high < 0)][["bins", "bin", "n_test2", "acc_test1", "acc_test2"]]
      .round(3).to_string(index=False))


# 4. Figure: classifier A, accuracy per bin on both test sets with 95% intervals, for each number of bins
BLUE, ORANGE = "#2a78d6", "#eb6834"
fig, axes = plt.subplots(1, len(BIN_COUNTS), figsize=(13, 3.8), sharey=True)
for ax, n_bins in zip(axes, BIN_COUNTS):
    d = detail[(detail.classifier == "A") & (detail.bins == n_bins)]
    for col, n, color, label, off in (("acc_test1", "n_test1", BLUE, "test set 1 (ImageNet val)", -0.12),
                                      ("acc_test2", "n_test2", ORANGE, "test set 2 (ImageNetV2)", 0.12)):
        err = 1.96 * np.sqrt(d[col] * (1 - d[col]) / d[n])
        ax.errorbar(d.bin + off, d[col], yerr=err, fmt="o-", ms=3, color=color, capsize=2, label=label)
    ax.set_title(f"{n_bins} bins")
    ax.set_xlabel("typicality bin (1 = least typical)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("top-1 accuracy (classifier A)")
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "fig_typicality_robustness.png", dpi=150, bbox_inches="tight")
plt.close()
