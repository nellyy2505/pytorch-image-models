# 07_generalisation_analysis.py - why accuracy drops on test set 2 (ImageNetV2) vs test set 1 (ImageNet val)
# --v2-images <folder holding imagenetv2-matched-frequency-format-val> also draws the misclassified images

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats

from common import PROC, OUT, SEED, N_CLASSES, MODELS, BLUE, ORANGE, GREY, load, class_sums, \
    typicality, typicality_label_free, read_predictions, kmeans_train_small, imagenet_info

plt.rcParams.update({"font.size": 11})

parser = argparse.ArgumentParser(description="A3: why accuracy drops on ImageNetV2")
parser.add_argument("--v2-images", type=Path, help="folder that holds imagenetv2-matched-frequency-format-val")
args = parser.parse_args()

CLASSIFIERS = ["linear", "mlp", "centroid"]
DRAWS = 1000
SHIFT_DRAWS = 200
BIN_COUNTS = (5, 10, 20)
ETA_REPEATS = 5
PCA_DIMS, N_CLUSTERS = 128, 50
rng = np.random.default_rng(SEED)


# 1. Data, class names and predictions
X, y = {}, {}
for t in ("test1", "test2"):
    X[t], y[t] = load(t)
n = {t: len(y[t]) for t in y}
info = imagenet_info()                             # ImageNet class names, shipped with timm
wnids = (info / "imagenet_synsets.txt").read_text().split()
lemmas = dict(line.split("\t", 1) for line in (info / "imagenet_synset_to_lemma.txt").read_text().splitlines())
class_name = np.array([lemmas[w].split(",")[0] for w in wnids])
meta1 = pd.read_csv(PROC / "test1_meta.csv")
if not (meta1.path.str.split("/").str[-2].to_numpy() == np.array(wnids)[y["test1"]]).all():
    raise ValueError("class names do not line up with the labels")

preds = {c: {s: read_predictions(c, s) for s in ("val", "test1", "test2")} for c in CLASSIFIERS}
correct = {c: {t: preds[c][t][:, 0] == y[t] for t in y} for c in CLASSIFIERS}
val_y = load("val")[1]
val_top1 = {c: (preds[c]["val"][:, 0] == val_y).mean() for c in CLASSIFIERS}
PRIMARY = max(val_top1, key=val_top1.get)           # final classifier: best on our validation set
print("validation top-1:", {c: round(v, 5) for c, v in val_top1.items()}, "-> primary:", MODELS[PRIMARY])

dup = pd.read_csv(OUT / "audit_test1_duplicates.csv").duplicate.to_numpy()
print(dup.sum(), "test set 1 images duplicate a training image")


# 2. Overall gap, with and without the duplicates
def gap_row(c, subset, c1, c2):
    p1, p2 = c1.mean(), c2.mean()
    se = np.sqrt(p1 * (1 - p1) / len(c1) + p2 * (1 - p2) / len(c2))
    return {"classifier": MODELS[c], "subset": subset, "test1_rows": len(c1), "test1_top1": p1, "test2_top1": p2,
            "gap": p1 - p2, "gap_ci_low": p1 - p2 - 1.96 * se, "gap_ci_high": p1 - p2 + 1.96 * se}


rows = []
for c in CLASSIFIERS:
    rows.append(gap_row(c, "all", correct[c]["test1"], correct[c]["test2"]))
    rows.append(gap_row(c, "test1 without duplicates", correct[c]["test1"][~dup], correct[c]["test2"]))
    rows[-2]["test1_top5"], rows[-2]["test2_top5"] = [(preds[c][t] == y[t][:, None]).any(axis=1).mean() for t in y]
    rows[-1]["duplicates_top1"] = correct[c]["test1"][dup].mean()
overall = pd.DataFrame(rows)
overall.to_csv(OUT / "gen_overall.csv", index=False)
print(overall.round(4).to_string(index=False))


# 3. Per-class gap: Fisher's exact test per class, Benjamini-Hochberg across classes
def per_class(c):
    k = {t: np.bincount(y[t], weights=correct[c][t], minlength=N_CLASSES) for t in y}
    m = {t: np.bincount(y[t], minlength=N_CLASSES) for t in y}
    p = np.array([stats.fisher_exact([[k["test1"][j], m["test1"][j] - k["test1"][j]],
                                      [k["test2"][j], m["test2"][j] - k["test2"][j]]])[1] for j in range(N_CLASSES)])
    return pd.DataFrame({"class": np.arange(N_CLASSES), "name": class_name, "acc_test1": k["test1"] / m["test1"],
                         "acc_test2": k["test2"] / m["test2"], "gap": k["test1"] / m["test1"] - k["test2"] / m["test2"],
                         "p_value": p, "q_value": stats.false_discovery_control(p)})


class_gaps = {c: per_class(c) for c in CLASSIFIERS}
for c in CLASSIFIERS:
    class_gaps[c].to_csv(OUT / f"gen_perclass_{c}.csv", index=False)
pc = class_gaps[PRIMARY]
pooled = (pc.acc_test1 * 50 + pc.acc_test2 * 10) / 60       # per-class gaps from sampling noise alone
null_gaps = rng.binomial(50, np.tile(pooled, 20)) / 50 - rng.binomial(10, np.tile(pooled, 20)) / 10
sig = pc.q_value < 0.05
print(f"\nper-class gap ({MODELS[PRIMARY]}): sd {pc.gap.std():.3f} observed vs {null_gaps.std():.3f} from sampling "
      f"noise; {sig.sum()} of {N_CLASSES} classes differ significantly (FDR 5%): "
      f"{(sig & (pc.gap > 0)).sum()} drops, {(sig & (pc.gap < 0)).sum()} rises")
worst = pc.sort_values(["q_value", "gap"], ascending=[True, False]).head(20)
print(worst[["class", "name", "acc_test1", "acc_test2", "gap", "q_value"]].round(3).to_string(index=False))

# most common wrong predictions on test set 2 for the 10 strongest drops
errors = []
for k in worst["class"].head(10):
    wrong = (y["test2"] == k) & ~correct[PRIMARY]["test2"]
    for pred, count in pd.Series(preds[PRIMARY]["test2"][wrong, 0]).value_counts().head(3).items():
        errors.append({"true_class": k, "true_name": class_name[k], "predicted_as": pred,
                       "predicted_name": class_name[pred], "count": count, "share_of_errors": count / wrong.sum()})
pd.DataFrame(errors).to_csv(OUT / "gen_error_patterns.csv", index=False)


# 4. Class centroid shift: 40 test set 1 images vs the 10 of test set 2, against the other 10 of test set 1
shift = []
for k in range(N_CLASSES):
    A, B = X["test1"][y["test1"] == k].astype(np.float64), X["test2"][y["test2"] == k].astype(np.float64)
    obs, ref = np.empty(SHIFT_DRAWS), np.empty(SHIFT_DRAWS)
    for d in range(SHIFT_DRAWS):
        s = rng.permutation(len(A))
        c40 = A[s[:40]].mean(axis=0)
        obs[d], ref[d] = np.linalg.norm(c40 - B.mean(axis=0)), np.linalg.norm(c40 - A[s[40:50]].mean(axis=0))
    shift.append({"class": k, "name": class_name[k], "observed": obs.mean(), "reference_mean": ref.mean(),
                  "reference_95th": np.percentile(ref, 95), "excess": obs.mean() - ref.mean()})
shift = pd.DataFrame(shift)
for c in CLASSIFIERS:
    shift[f"gap_{c}"] = class_gaps[c].gap
shift.to_csv(OUT / "gen_centroid_shift.csv", index=False)
print(f"\ncentroid distance: observed {shift.observed.mean():.2f}, reference {shift.reference_mean.mean():.2f}; "
      f"above the reference mean in {(shift.observed > shift.reference_mean).mean():.1%} of classes, above its 95th "
      f"percentile in {(shift.observed > shift.reference_95th).mean():.1%}")
for c in CLASSIFIERS:
    print(f"  {MODELS[c]}: excess shift vs class gap  Pearson {stats.pearsonr(shift.excess, shift[f'gap_{c}'])[0]:.3f}"
          f"  Spearman {stats.spearmanr(shift.excess, shift[f'gap_{c}'])[0]:.3f}")


# within- and between-class variance; test set 1 cut to 10 per class to match test set 2
def ten_per_class():
    return np.concatenate([rng.choice(np.flatnonzero(y["test1"] == k), 10, replace=False) for k in range(N_CLASSES)])


def decompose(Z, labels):
    Z = Z.astype(np.float64)
    means = class_sums(Z, labels) / np.bincount(labels, minlength=N_CLASSES)[:, None]
    between = (np.bincount(labels, minlength=N_CLASSES)[:, None] * (means - Z.mean(axis=0)) ** 2).sum() / len(Z)
    return ((Z - means[labels]) ** 2).sum() / len(Z), between


w1, b1 = np.mean([decompose(X["test1"][r], y["test1"][r]) for r in (ten_per_class() for _ in range(ETA_REPEATS))],
                 axis=0)
w2, b2 = decompose(X["test2"], y["test2"])
variance = pd.DataFrame({"test_set": ["test1 (10 per class)", "test2"], "within_class": [w1, w2],
                         "between_class": [b1, b2]})
variance["between_share"] = variance.between_class / (variance.within_class + variance.between_class)
variance.to_csv(OUT / "gen_variance.csv", index=False)
print(variance.round(3).to_string(index=False))
print(f"within-class variance {100 * (w2 / w1 - 1):+.1f}%, between-class {100 * (b2 / b1 - 1):+.1f}% on test set 2")


# 5. Typicality: test set 2's accuracy predicted from its typicality mix, with and without the true label
measures = {"own label": {t: typicality(X[t], y[t]) for t in y},
            "label-free": {t: typicality_label_free(X[t]) for t in y}}


def predicted_test2(b1_, b2_, c1, n_bins):
    n1 = np.bincount(b1_, minlength=n_bins)
    acc1 = np.bincount(b1_, weights=c1, minlength=n_bins) / np.maximum(n1, 1)
    return (np.bincount(b2_, minlength=n_bins) / len(b2_) * acc1)[n1 > 0].sum()


summary, per_bin = [], []
for m, typ in measures.items():
    for n_bins in BIN_COUNTS:
        edges = np.quantile(typ["test1"], np.linspace(0, 1, n_bins + 1))[1:-1]
        bins = {t: np.digitize(typ[t], edges) for t in y}
        for c in CLASSIFIERS:
            c1, c2 = correct[c]["test1"].astype(float), correct[c]["test2"].astype(float)
            pred = predicted_test2(bins["test1"], bins["test2"], c1, n_bins)
            boot = []
            for _ in range(DRAWS):
                i, j = rng.integers(0, n["test1"], n["test1"]), rng.integers(0, n["test2"], n["test2"])
                boot.append(c2[j].mean() - predicted_test2(bins["test1"][i], bins["test2"][j], c1[i], n_bins))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            summary.append({"measure": m, "bins": n_bins, "classifier": MODELS[c], "test1_top1": c1.mean(),
                            "test2_top1": c2.mean(), "test2_predicted": pred, "residual": c2.mean() - pred,
                            "residual_ci_low": lo, "residual_ci_high": hi,
                            "share_of_gap_reproduced": (c1.mean() - pred) / (c1.mean() - c2.mean())})
            for b in range(n_bins):
                m1, m2 = bins["test1"] == b, bins["test2"] == b
                per_bin.append({"measure": m, "bins": n_bins, "classifier": MODELS[c], "bin": b + 1,
                                "share_test1": m1.mean(), "share_test2": m2.mean(), "acc_test1": c1[m1].mean(),
                                "acc_test2": c2[m2].mean() if m2.any() else np.nan, "n_test2": int(m2.sum())})
summary, per_bin = pd.DataFrame(summary), pd.DataFrame(per_bin)
summary.to_csv(OUT / "gen_typicality.csv", index=False)
per_bin.to_csv(OUT / "gen_typicality_bins.csv", index=False)
print(f"\nmean typicality (own label): test1 {measures['own label']['test1'].mean():.3f}, "
      f"test2 {measures['own label']['test2'].mean():.3f}")
print(summary[summary.classifier == MODELS[PRIMARY]][["measure", "bins", "test2_top1", "test2_predicted",
      "residual_ci_low", "residual_ci_high", "share_of_gap_reproduced"]].round(4).to_string(index=False))


# 6. Per feature: mean shift and class separability
X1, X2 = X["test1"], X["test2"]
_, p_feat = stats.ttest_ind(X2, X1, axis=0, equal_var=False)
d = (X2.mean(axis=0) - X1.mean(axis=0)) / np.sqrt((X1.var(axis=0, ddof=1) + X2.var(axis=0, ddof=1)) / 2)


def eta_squared(Z, labels):
    Z = Z.astype(np.float64)
    counts = np.bincount(labels, minlength=N_CLASSES)
    means = class_sums(Z, labels) / counts[:, None]
    return (counts[:, None] * (means - Z.mean(axis=0)) ** 2).sum(axis=0) / len(Z) / Z.var(axis=0)


X_small, y_small = load("train_small")
eta_train, eta_test2 = eta_squared(X_small, y_small), eta_squared(X2, y["test2"])
eta_test1 = np.mean([eta_squared(X1[r], y["test1"][r]) for r in (ten_per_class() for _ in range(ETA_REPEATS))],
                    axis=0)
features = pd.DataFrame({"feature": np.arange(X1.shape[1]), "cohens_d": d, "p_value": p_feat,
                         "q_value": stats.false_discovery_control(p_feat), "eta2_train": eta_train,
                         "eta2_test1": eta_test1, "eta2_test2": eta_test2, "eta2_drop": eta_test1 - eta_test2})
features.to_csv(OUT / "gen_features.csv", index=False)
print(f"\nper feature: {(features.q_value < 0.05).sum()} of {len(features)} shift significantly (FDR 5%), "
      f"|d| median {np.median(np.abs(d)):.3f}, max {np.abs(d).max():.3f}; separability test1 {eta_test1.mean():.3f}, "
      f"test2 {eta_test2.mean():.3f}, lower on test2 for {(eta_test2 < eta_test1).mean():.1%} of features")
print(f"Spearman with training separability: |d| {stats.spearmanr(np.abs(d), eta_train)[0]:.3f}, "
      f"separability drop {stats.spearmanr(features.eta2_drop, eta_train)[0]:.3f}")


# 7. k-means clusters of train_small; shares compared within class (images outside their class's usual cluster)
pca, kmeans, assigned = kmeans_train_small(X_small, N_CLUSTERS, [X[t] for t in y], PCA_DIMS)
cluster = dict(zip(y, assigned))
usual = pd.Series(kmeans.labels_).groupby(y_small).agg(lambda s: s.value_counts().index[0]).to_numpy()
outside = {t: cluster[t] != usual[y[t]] for t in y}
table = [[outside[t].sum(), (~outside[t]).sum()] for t in y]
chi2, p_out = stats.chi2_contingency(table)[:2]
print(f"\nimages outside their class's usual cluster: test1 {outside['test1'].mean():.1%}, "
      f"test2 {outside['test2'].mean():.1%} (chi-squared {chi2:.0f}, p = {p_out:.2g})")
pd.DataFrame({"test_set": list(y), "images": [n[t] for t in y], "outside": [int(outside[t].sum()) for t in y],
              "share_outside": [outside[t].mean() for t in y], "chi2": chi2, "p_value": p_out}
             ).to_csv(OUT / "gen_cluster_outside.csv", index=False)

clusters = []
for k in range(N_CLUSTERS):
    m1, m2 = cluster["test1"] == k, cluster["test2"] == k
    a1, a2 = correct[PRIMARY]["test1"][m1], correct[PRIMARY]["test2"][m2]
    names = class_name[pd.Series(y_small[kmeans.labels_ == k]).value_counts().head(3).index]
    clusters.append({"cluster": k, "example_classes": ", ".join(names), "n_test1": int(m1.sum()),
                     "n_test2": int(m2.sum()), "acc_test1": a1.mean(), "acc_test2": a2.mean() if m2.any() else np.nan,
                     "p_value": stats.fisher_exact([[a1.sum(), (~a1).sum()], [a2.sum(), (~a2).sum()]])[1]
                     if m2.any() else 1.0,
                     "gap_contribution": a1.sum() / n["test1"] - a2.sum() / n["test2"]})
clusters = pd.DataFrame(clusters)
clusters["gap"] = clusters.acc_test1 - clusters.acc_test2
clusters["q_value"] = stats.false_discovery_control(clusters.p_value)
clusters["gap_contribution"] /= clusters.gap_contribution.sum()
clusters.sort_values("gap", ascending=False).to_csv(OUT / "gen_clusters.csv", index=False)
sig_c = clusters.q_value < 0.05
print(f"{sig_c.sum()} of {N_CLUSTERS} clusters differ significantly ({(sig_c & (clusters.gap > 0)).sum()} drops), "
      f"together {clusters.gap_contribution[sig_c].sum():.0%} of the gap; test set 2 images per cluster "
      f"{clusters.n_test2.min()} to {clusters.n_test2.max()} (median {clusters.n_test2.median():.0f})")


# 8. Figures
def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


main = overall[overall.subset == "all"]
fig, ax = plt.subplots(figsize=(7.5, 4))
x = np.arange(len(main))
for off, t, color in ((-0.175, "test1", BLUE), (0.175, "test2", ORANGE)):
    acc = main[f"{t}_top1"].to_numpy()
    ax.bar(x + off, acc, 0.35, color=color, label=f"test set {t[-1]}", yerr=1.96 * np.sqrt(acc * (1 - acc) / n[t]),
           capsize=3)
    for i, a in enumerate(acc):
        ax.text(i + off, a + 0.025, f"{100 * a:.1f}%", ha="center", fontsize=9)
ax.set_xticks(x, main.classifier)
ax.set_ylabel("top-1 accuracy")
ax.set_ylim(0, 1.15)
ax.legend(frameon=False, loc="upper center", ncol=2)
ax.set_title("Top-1 on ImageNet val (test set 1) and ImageNetV2 (test set 2), 95% intervals")
tidy(ax)
plt.savefig(OUT / "fig_gen_overall.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
bins_ = np.linspace(-1, 1, 41)
axes[0].hist(pc.gap, bins=bins_, color=BLUE, alpha=0.85, edgecolor="white", label="observed")
axes[0].hist(null_gaps, bins=bins_, weights=np.full(len(null_gaps), 1 / 20), histtype="step", color="k",
             label="sampling noise only")
axes[0].set_xlabel("per-class top-1 gap (test1 - test2)")
axes[0].set_ylabel("number of classes")
axes[0].set_title(f"Per-class gaps, {MODELS[PRIMARY]}")
axes[0].legend(frameon=False)
jitter = rng.uniform(-0.015, 0.015, N_CLASSES)        # gaps come in steps, so points would stack
for mask, color, label in ((~sig, BLUE, "not significant"), (sig, ORANGE, "significant (FDR 5%)")):
    axes[1].scatter(shift.excess[mask], pc.gap[mask] + jitter[mask], s=12, alpha=0.6, color=color, label=label)
axes[1].axvline(0, color=GREY, lw=1)
axes[1].set_xlabel("centroid shift beyond the test set 1 reference")
axes[1].set_ylabel("per-class top-1 gap")
axes[1].set_title("Centroid shift vs class gap")
axes[1].legend(frameon=False, loc="lower right")
for ax in axes:
    tidy(ax)
plt.tight_layout()
plt.savefig(OUT / "fig_gen_class_gaps.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
own = measures["own label"]
sim_bins = np.linspace(min(own["test1"].min(), own["test2"].min()), 1, 50)
for t, color in (("test1", BLUE), ("test2", ORANGE)):
    axes[0].hist(own[t], bins=sim_bins, density=True, histtype="step", lw=1.6, color=color,
                 label=f"test set {t[-1]} (mean {own[t].mean():.3f})")
axes[0].set_xlabel("cosine similarity to own-class training centroid")
axes[0].set_ylabel("density")
axes[0].set_title("Typicality (own label)")
for ax, m in zip(axes[1:], measures):
    part = per_bin[(per_bin.measure == m) & (per_bin.bins == 10) & (per_bin.classifier == MODELS[PRIMARY])]
    share = summary[(summary.measure == m) & (summary.bins == 10) & (summary.classifier == MODELS[PRIMARY])]
    for t, color in (("test1", BLUE), ("test2", ORANGE)):
        acc, cnt = part[f"acc_{t}"], part["n_test2"] if t == "test2" else n["test1"] / 10
        ax.errorbar(part.bin, acc, yerr=1.96 * np.sqrt(acc * (1 - acc) / cnt), fmt="o-", ms=3, capsize=2,
                    color=color, label=f"test set {t[-1]}")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel(f"{m} typicality decile of test set 1 (1 = least typical)")
    ax.set_ylabel(f"top-1, {MODELS[PRIMARY]}")
    ax.set_title(f"{m}: reproduces {share.share_of_gap_reproduced.iloc[0]:.0%} of the gap")
    ax.legend(frameon=False, loc="lower right")
for ax in axes:
    tidy(ax)
plt.tight_layout()
plt.savefig(OUT / "fig_gen_typicality.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(d, bins=50, color=BLUE, edgecolor="white")
axes[0].axvline(0, color=GREY, lw=1)
axes[0].set_xlabel("Cohen's d per feature (test2 - test1)")
axes[0].set_ylabel("number of features")
axes[0].set_title(f"Mean shift: {(features.q_value < 0.05).sum()} of {len(d)} significant")
lims = [0, max(eta_test1.max(), eta_test2.max()) * 1.05]
sc = axes[1].scatter(eta_test1, eta_test2, c=eta_train, s=10, cmap="viridis")
axes[1].plot(lims, lims, "--", color=GREY)
axes[1].set_xlim(lims)
axes[1].set_ylim(lims)
axes[1].set_xlabel("class separability on test set 1 (eta squared)")
axes[1].set_ylabel("class separability on test set 2")
axes[1].set_title("Class separability per feature")
fig.colorbar(sc, ax=axes[1], label="separability on train_small")
axes[2].bar([0, 1], variance.within_class, 0.4, color=BLUE, label="within-class")
axes[2].bar([0.4, 1.4], variance.between_class, 0.4, color=ORANGE, label="between-class")
axes[2].set_xticks([0.2, 1.2], ["test set 1\n(10 per class)", "test set 2"])
axes[2].set_ylabel("variance, summed over features")
axes[2].set_title("Variance decomposition")
axes[2].legend(frameon=False)
for ax in axes:
    tidy(ax)
plt.tight_layout()
plt.savefig(OUT / "fig_gen_features.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for mask, color, label in ((~sig_c, BLUE, "gap not significant"), (sig_c, ORANGE, "significant gap (FDR 5%)")):
    ax.scatter(clusters.acc_test1[mask], clusters.acc_test2[mask], s=2 * clusters.n_test2[mask], alpha=0.6,
               color=color, label=label)
lo = min(clusters.acc_test1.min(), clusters.acc_test2.min()) - 0.03
ax.plot([lo, 1], [lo, 1], "--", color=GREY)
for _, r in clusters.sort_values("gap", ascending=False).head(4).iterrows():
    ax.annotate(r.example_classes.split(",")[0], (r.acc_test1, r.acc_test2), fontsize=8, xytext=(4, -10),
                textcoords="offset points")
ax.set_xlabel("top-1 on test set 1")
ax.set_ylabel("top-1 on test set 2")
ax.set_title(f"k-means clusters ({MODELS[PRIMARY]}); size = test set 2 images")
ax.legend(frameon=False, loc="upper left", markerscale=0.3)
tidy(ax)
plt.savefig(OUT / "fig_gen_clusters.png", dpi=150, bbox_inches="tight")
plt.close()


# 9. Misclassified ImageNetV2 images of the classes with the strongest drop
if args.v2_images is None:
    print("\nno --v2-images given; skipping fig_gen_v2_errors.png")
else:
    meta2 = pd.read_csv(PROC / "test2_meta.csv")
    show = worst["class"].head(6).to_numpy()
    fig, axes = plt.subplots(len(show), 5, figsize=(12, 2.6 * len(show)))
    for row, k in zip(axes, show):
        for ax in row:
            ax.axis("off")
        for ax, i in zip(row, np.flatnonzero((y["test2"] == k) & ~correct[PRIMARY]["test2"])[:5]):
            ax.imshow(Image.open(args.v2_images / meta2.path[i]).convert("RGB"))
            ax.set_title(f"predicted: {class_name[preds[PRIMARY]['test2'][i, 0]]}", fontsize=8)
        row[0].text(-0.08, 0.5, class_name[k], transform=row[0].transAxes, ha="right", va="center", weight="bold")
    fig.tight_layout()
    fig.suptitle(f"Misclassified ImageNetV2 images ({MODELS[PRIMARY]}), classes with the strongest drop", y=1.01)
    plt.savefig(OUT / "fig_gen_v2_errors.png", dpi=110, bbox_inches="tight")
    plt.close()
