# 06_nearest_centroid.py - classifier C: nearest class centroid on EVA-02 features

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from common import OUT, SEED, N_CLASSES, CHUNK, SETS, BLUE, ORANGE, GREY, load, training_split, class_sums, unit, \
    accuracy, save_predictions, centroid_scores

plt.rcParams.update({"font.size": 11})

PCA_DIMS = [32, 64, 128, 256, 512, 768, 1024]
SHRINKAGE = [0.01, 0.03, 0.1, 0.3, 1.0]
PER_CLASS = [2, 5, 10, 25, 50, 100]


# 1. Load training data and our validation set
X_train, y_train = load("train_small")
X_val, y_val = load("val")
X_full, y_full, train_idx = training_split()


# 2. Nearest centroid
def fit_centroids(Z, y, cosine):
    if cosine:
        Z = unit(Z)
    C = class_sums(Z, y) / np.bincount(y, minlength=N_CLASSES)[:, None]
    return unit(C) if cosine else C


def predict_top5(Z, C, cosine):
    return np.vstack([np.argsort(-centroid_scores(Z[i:i + CHUNK], C, cosine), axis=1)[:, :5]
                      for i in range(0, len(Z), CHUNK)])


# 3. Variants: a transform fitted on train_small + Euclidean or cosine distance, scored on our validation set
experiments, variants = [], {}


def run(experiment, name, transform, cosine):
    t0 = time.time()
    C = fit_centroids(transform(X_train), y_train, cosine)
    top1, top5 = accuracy(predict_top5(transform(X_val), C, cosine), y_val)
    experiments.append({"experiment": experiment, "model": name, "val_top1": top1, "val_top5": top5,
                        "seconds": time.time() - t0})
    variants[name] = (transform, cosine)
    print("%s  %-36s val top-1 %.4f  top-5 %.4f" % (experiment, name, top1, top5), flush=True)


mean, std = X_train.mean(axis=0), X_train.std(axis=0)

# E1 normalisation
run("E1", "raw, Euclidean", lambda X: X, cosine=False)
run("E1", "standardised, Euclidean", lambda X: (X - mean) / std, cosine=False)
run("E1", "raw, cosine", lambda X: X, cosine=True)
run("E1", "centred, cosine", lambda X: X - mean, cosine=True)
run("E1", "standardised, cosine", lambda X: (X - mean) / std, cosine=True)

# E2 PCA, with and without whitening, using the better distance from E1
e1 = pd.DataFrame(experiments)
use_cosine = "cosine" in e1.loc[e1.val_top1.idxmax(), "model"]
for dims in PCA_DIMS:
    for whiten in (False, True):
        pca = PCA(n_components=dims, whiten=whiten, random_state=SEED).fit(X_train)
        run("E2", "PCA-%d%s" % (dims, " whitened" if whiten else ""), pca.transform, use_cosine)

# E3 LDA with shrinkage (1.0 = plain Euclidean)
for shrinkage in SHRINKAGE:
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrinkage).fit(X_train, y_train)
    for cosine in (False, True):
        name = "LDA shrinkage %.2f, %s" % (shrinkage, "cosine" if cosine else "Euclidean")
        run("E3", name, lambda X, W=lda.scalings_: (X - mean) @ W, cosine)

experiments = pd.DataFrame(experiments)
experiments.to_csv(OUT / "centroid_experiments.csv", index=False)

# choose the simplest variant (first in the table) within one standard error of the best
top = experiments.val_top1.max()
std_error = np.sqrt(top * (1 - top) / len(y_val))
chosen = experiments[experiments.val_top1 >= top - std_error].model.iloc[0]
transform, cosine = variants[chosen]
print("best val top-1 %.4f, standard error %.4f; chosen: %s" % (top, std_error, chosen))


# 4. E4 learning curve, then the chosen variant on the full training split
rng = np.random.default_rng(SEED)
order = [rng.permutation(np.flatnonzero(y_train == c)) for c in range(N_CLASSES)]
curve = []
for n in PER_CLASS:
    rows = np.concatenate([o[:n] for o in order])
    C = fit_centroids(transform(X_train[rows]), y_train[rows], cosine)
    curve.append({"per_class": n, "val_top1": accuracy(predict_top5(transform(X_val), C, cosine), y_val)[0]})

# the transform stays fitted on train_small; the centroids use the full training split
t0 = time.time()
sums, counts = None, np.zeros(N_CLASSES)
for i in range(0, len(train_idx), CHUNK):
    rows = train_idx[i:i + CHUNK]
    Z = transform(np.asarray(X_full[rows]))
    s = class_sums(unit(Z) if cosine else Z, y_full[rows])
    sums = s if sums is None else sums + s
    counts += np.bincount(y_full[rows], minlength=N_CLASSES)
C = sums / counts[:, None]
C = unit(C) if cosine else C
np.savez(OUT / "centroid_model.npz", centroids=C, cosine=cosine, model=chosen)     # for 08b
final = {"model": chosen, "train_rows": len(train_idx), "train_seconds": time.time() - t0}


# 5. Score once on both test sets
for s in SETS:
    X, y = load(s)
    top5 = predict_top5(transform(X), C, cosine)
    save_predictions("centroid", s, y, top5)
    final[f"{s}_top1"], final[f"{s}_top5"] = accuracy(top5, y)
curve.append({"per_class": len(train_idx) / N_CLASSES, "val_top1": final["val_top1"]})
curve = pd.DataFrame(curve)
curve.to_csv(OUT / "centroid_learning_curve.csv", index=False)
pd.DataFrame([final]).to_csv(OUT / "centroid_final.csv", index=False)
print(pd.Series(final).to_string())


# 6. Figures
e1 = experiments[experiments.experiment == "E1"]
plt.figure(figsize=(6.2, 3))
plt.barh(e1.model, e1.val_top1, color=BLUE)
for i, v in enumerate(e1.val_top1):
    plt.text(v, i, " %.2f%%" % (100 * v), va="center")
plt.gca().invert_yaxis()
plt.xlim(e1.val_top1.min() - 0.05, e1.val_top1.max() + 0.04)
plt.title("E1: feature normalisation (axis does not start at 0)")
plt.xlabel("validation top-1 accuracy")
plt.savefig(OUT / "fig_centroid_normalisation.png", dpi=150, bbox_inches="tight")
plt.close()

e2 = experiments[experiments.experiment == "E2"]
plt.figure(figsize=(6.2, 3.2))
plt.plot(PCA_DIMS, e2[~e2.model.str.contains("whitened")].val_top1, "o-", label="PCA", color=BLUE)
plt.plot(PCA_DIMS, e2[e2.model.str.contains("whitened")].val_top1, "o-", label="PCA + whitening", color=ORANGE)
plt.axhline(e1.val_top1.max(), ls="--", color=GREY, label="best without PCA (E1)")
plt.xscale("log", base=2)
plt.xticks(PCA_DIMS, PCA_DIMS)
plt.title("E2: number of principal components")
plt.xlabel("principal components kept")
plt.ylabel("validation top-1 accuracy")
plt.legend(frameon=False)
plt.savefig(OUT / "fig_centroid_pca.png", dpi=150, bbox_inches="tight")
plt.close()

e3 = experiments[experiments.experiment == "E3"]
plt.figure(figsize=(6.2, 3.2))
plt.plot(SHRINKAGE, e3[e3.model.str.endswith("Euclidean")].val_top1, "o-", label="Euclidean", color=BLUE)
plt.plot(SHRINKAGE, e3[e3.model.str.endswith("cosine")].val_top1, "o-", label="cosine", color=ORANGE)
plt.xscale("log")
plt.title("E3: LDA, shrinkage of the shared covariance")
plt.xlabel("shrinkage (1 = identity, i.e. no LDA)")
plt.ylabel("validation top-1 accuracy")
plt.legend(frameon=False)
plt.savefig(OUT / "fig_centroid_lda.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6.2, 3.2))
plt.plot(curve.per_class[:-1], curve.val_top1[:-1], "o-", color=BLUE, label="train_small subsets")
plt.plot(curve.per_class.iloc[-1:], curve.val_top1.iloc[-1:], "s", color=ORANGE,
         label="full training split (%.0f per class on average)" % curve.per_class.iloc[-1])
plt.xscale("log")
plt.title("E4: learning curve (%s)" % chosen)
plt.xlabel("training images per class")
plt.ylabel("validation top-1 accuracy")
plt.legend(frameon=False)
plt.savefig(OUT / "fig_centroid_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
