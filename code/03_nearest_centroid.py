# 03_nearest_centroid.py - Classifier C: nearest-centroid classifier on the EVA-02 features (1,024-d)
#
# Each class is represented by the mean (centroid) of its training features.
# A new image gets the class of the closest centroid.
# Variants are compared on our validation set (train_small -> val):
#   E1 feature normalisation, E2 PCA, E3 LDA (shared within-class covariance), E4 learning curve.
# The chosen variant (the simplest one within one standard error of the best) and the raw baseline
# are then trained on the full training set and scored once on test set 1 (ImageNet val) and test set 2 (ImageNetV2).

# import packages
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

plt.rcParams.update({"font.size": 11})

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 7
N_CLASSES = 1000
CHUNK = 20_000            # rows per mini-batch when we go through big arrays
PCA_DIMS = [32, 64, 128, 256, 512, 768, 1024]
SHRINKAGE = [0.01, 0.03, 0.1, 0.3, 1.0]
PER_CLASS = [2, 5, 10, 25, 50, 100]


# 1. Load the data
# only training data and our validation set here; the test sets are loaded in section 5
X_train = np.load(PROC / "train_small_X.npy")
y_train = np.load(PROC / "train_small_y.npy").astype(int)
X_val = np.load(PROC / "val_X.npy")
y_val = np.load(PROC / "val_y.npy").astype(int)
print("X_train shape", X_train.shape)
print("X_val shape", X_val.shape)

# the full training set is 5 GB, so it is memory-mapped and read in chunks
X_full = np.load(PROC / "train_full_X.npy", mmap_mode="r")
y_full = np.load(PROC / "train_full_y.npy").astype(int)
train_idx = np.load(PROC / "split_train_idx.npy")      # all training rows except our validation rows
print("X_full shape", X_full.shape, "- rows used for training:", len(train_idx))


# 2. Nearest-centroid functions
def l2_normalise(Z):
    # scale every row to length 1, so the dot product becomes the cosine similarity
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def fit_centroids(Z, y, cosine):
    # centroid = mean feature vector of each class
    if cosine:
        Z = l2_normalise(Z)
    C = np.zeros((N_CLASSES, Z.shape[1]))
    np.add.at(C, y, Z)
    C = C / np.bincount(y, minlength=N_CLASSES)[:, None]
    return l2_normalise(C) if cosine else C


def predict_top5(Z, C, cosine):
    # score every class and keep the 5 best, best first
    # cosine: score = cosine similarity; Euclidean: score = -|z - c|^2 / 2 (the |z|^2 part is the same for all classes)
    top5 = []
    for i in range(0, len(Z), CHUNK):
        Zb = Z[i:i + CHUNK]
        if cosine:
            scores = l2_normalise(Zb) @ C.T
        else:
            scores = Zb @ C.T - 0.5 * (C ** 2).sum(axis=1)
        top5.append(np.argsort(-scores, axis=1)[:, :5])
    return np.vstack(top5)


def accuracy(top5, y):
    top1_acc = (top5[:, 0] == y).mean()
    top5_acc = (top5 == y[:, None]).any(axis=1).mean()
    return top1_acc, top5_acc


# 3. Try the variants on our validation set
# each variant = a feature transform (fitted on train_small only) + Euclidean or cosine distance
experiments = []
variants = {}


def run(experiment, name, transform, cosine):
    t0 = time.time()
    C = fit_centroids(transform(X_train), y_train, cosine)
    fit_s = time.time() - t0
    t0 = time.time()
    top1, top5 = accuracy(predict_top5(transform(X_val), C, cosine), y_val)
    predict_s = time.time() - t0
    experiments.append({"experiment": experiment, "model": name, "val_top1": top1, "val_top5": top5,
                        "fit_s": fit_s, "predict_s": predict_s})
    variants[name] = (transform, cosine)
    print("%s  %-36s val top-1 %.4f  top-5 %.4f" % (experiment, name, top1, top5))


# mean and standard deviation of every feature, from the training data
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

# E1 normalisation
run("E1", "raw, Euclidean", lambda X: X, cosine=False)
run("E1", "standardised, Euclidean", lambda X: (X - mean) / std, cosine=False)
run("E1", "raw, cosine", lambda X: X, cosine=True)
run("E1", "centred, cosine", lambda X: X - mean, cosine=True)
run("E1", "standardised, cosine", lambda X: (X - mean) / std, cosine=True)

# E2 PCA: keep the top principal components, with and without whitening (each component scaled to variance 1)
# the distance (cosine or Euclidean) is the better one from E1
e1 = pd.DataFrame(experiments)
use_cosine = "cosine" in e1.loc[e1.val_top1.idxmax(), "model"]
for dims in PCA_DIMS:
    for whiten in (False, True):
        pca = PCA(n_components=dims, whiten=whiten, random_state=SEED).fit(X_train)
        name = "PCA-%d%s" % (dims, " whitened" if whiten else "")
        run("E2", name, pca.transform, use_cosine)

# E3 LDA: measure distances after removing the within-class covariance shared by all classes
# (Mahalanobis distance); shrinkage blends that covariance with the identity, 1.0 = plain Euclidean again
for shrinkage in SHRINKAGE:
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrinkage).fit(X_train, y_train)
    for cosine in (False, True):
        name = "LDA shrinkage %.2f, %s" % (shrinkage, "cosine" if cosine else "Euclidean")
        run("E3", name, lambda X, W=lda.scalings_: (X - mean) @ W, cosine)

experiments = pd.DataFrame(experiments)
experiments.to_csv(OUT / "ncm_experiments.csv", index=False)

# choose the model: the simplest variant within one standard error of the best val accuracy
# (smaller differences are just noise from which 50,000 images happen to be in val)
# the variants were run from simplest (raw, Euclidean) to most complex, so the first one in the table wins
# this is conservative: two models scored on the same images differ by less noise than one accuracy's standard error
top = experiments.val_top1.max()
std_error = np.sqrt(top * (1 - top) / len(y_val))
good_enough = experiments[experiments.val_top1 >= top - std_error]
best_name = good_enough.model.iloc[0]
best_transform, best_cosine = variants[best_name]
print("highest val top-1 %.4f (%s), standard error %.4f" % (top, experiments.loc[experiments.val_top1.idxmax(), "model"],
                                                            std_error))
print("chosen model (simplest within one standard error):", best_name)

# the models trained on the full training set: the baseline and the chosen one (the same model if it is the baseline)
models = list(dict.fromkeys(["raw, Euclidean", best_name]))

# E4 learning curve: how many training images per class does a centroid need?
rng = np.random.default_rng(SEED)
order = [rng.permutation(np.flatnonzero(y_train == c)) for c in range(N_CLASSES)]
curve = []
for n in PER_CLASS:
    rows = np.concatenate([o[:n] for o in order])
    for name in models:
        transform, cosine = variants[name]
        C = fit_centroids(transform(X_train[rows]), y_train[rows], cosine)
        top1, top5 = accuracy(predict_top5(transform(X_val), C, cosine), y_val)
        curve.append({"model": name, "per_class": n, "val_top1": top1})
        print("E4  %-36s %3d per class: val top-1 %.4f" % (name, n, top1))


# 4. Train the baseline and the chosen model on the full training set
# the transform stays the one fitted on train_small; only the centroids use all 1.23 M rows
def fit_centroids_full(transform, cosine):
    sums, counts = None, np.zeros(N_CLASSES)
    for i in range(0, len(train_idx), CHUNK):
        rows = train_idx[i:i + CHUNK]
        Z = transform(np.asarray(X_full[rows]))
        if cosine:
            Z = l2_normalise(Z)
        if sums is None:
            sums = np.zeros((N_CLASSES, Z.shape[1]))
        np.add.at(sums, y_full[rows], Z)
        counts += np.bincount(y_full[rows], minlength=N_CLASSES)
    C = sums / counts[:, None]
    return l2_normalise(C) if cosine else C


final = {}
for name in models:
    transform, cosine = variants[name]
    t0 = time.time()
    final[name] = fit_centroids_full(transform, cosine)
    print("%s trained on the full training set in %.0f s" % (name, time.time() - t0))


# 5. Score once on both test sets (never used before this point)
X_test1 = np.load(PROC / "test1_X.npy")
y_test1 = np.load(PROC / "test1_y.npy").astype(int)
X_test2 = np.load(PROC / "test2_X.npy")
y_test2 = np.load(PROC / "test2_y.npy").astype(int)
print("X_test1 shape", X_test1.shape, "X_test2 shape", X_test2.shape)
sets = {"val": (X_val, y_val), "test1": (X_test1, y_test1), "test2": (X_test2, y_test2)}

results = []
for name, C in final.items():
    transform, cosine = variants[name]
    for s, (X, y) in sets.items():
        t0 = time.time()
        top5 = predict_top5(transform(X), C, cosine)
        top1_acc, top5_acc = accuracy(top5, y)
        results.append({"model": name, "set": s, "rows": len(y), "top1": top1_acc, "top5": top5_acc,
                        "predict_s": time.time() - t0})
        # save the top-5 predictions of the chosen model, row by row (same order as data/processed/<set>_meta.csv)
        if name == best_name:
            pred = pd.DataFrame(top5, columns=["pred1", "pred2", "pred3", "pred4", "pred5"])
            pred.insert(0, "label", y)
            pred.to_csv(OUT / ("ncm_predictions_%s.csv" % s), index_label="row")
    curve.append({"model": name, "per_class": len(train_idx) // N_CLASSES, "val_top1": results[-3]["top1"]})

results = pd.DataFrame(results)
results.to_csv(OUT / "ncm_results.csv", index=False)
curve = pd.DataFrame(curve)
curve.to_csv(OUT / "ncm_learning_curve.csv", index=False)
print(results.round(4).to_string(index=False))
for name in final:
    r = results[results.model == name].set_index("set")
    print("%s: top-1 gap (test set 1 - test set 2) = %.1f points" % (name, 100 * (r.top1["test1"] - r.top1["test2"])))


# 6. Figures
# E1 normalisation
e1 = experiments[experiments.experiment == "E1"]
plt.figure(figsize=(6.2, 3))
plt.barh(e1.model, e1.val_top1, color="#2a78d6")
for i, v in enumerate(e1.val_top1):
    plt.text(v, i, " %.2f%%" % (100 * v), va="center")
plt.gca().invert_yaxis()
plt.xlim(e1.val_top1.min() - 0.05, e1.val_top1.max() + 0.04)
plt.title("E1: feature normalisation")
plt.xlabel("validation top-1 accuracy")
plt.savefig(OUT / "fig_ncm_normalisation.png", dpi=150, bbox_inches="tight")
plt.close()

# E2 PCA
e2 = experiments[experiments.experiment == "E2"]
plain = e2[~e2.model.str.contains("whitened")]
whitened = e2[e2.model.str.contains("whitened")]
plt.figure(figsize=(6.2, 3.2))
plt.plot(PCA_DIMS, plain.val_top1, "o-", label="PCA", color="#2a78d6")
plt.plot(PCA_DIMS, whitened.val_top1, "o-", label="PCA + whitening", color="#eb6834")
plt.axhline(e1.val_top1.max(), ls="--", color="grey", label="best without PCA (E1)")
plt.xscale("log", base=2)
plt.xticks(PCA_DIMS, PCA_DIMS)
plt.title("E2: number of principal components")
plt.xlabel("principal components kept")
plt.ylabel("validation top-1 accuracy")
plt.legend(frameon=False)
plt.savefig(OUT / "fig_ncm_pca.png", dpi=150, bbox_inches="tight")
plt.close()

# E3 LDA shrinkage
e3 = experiments[experiments.experiment == "E3"]
plt.figure(figsize=(6.2, 3.2))
plt.plot(SHRINKAGE, e3[e3.model.str.endswith("Euclidean")].val_top1, "o-", label="Euclidean", color="#2a78d6")
plt.plot(SHRINKAGE, e3[e3.model.str.endswith("cosine")].val_top1, "o-", label="cosine", color="#eb6834")
plt.xscale("log")
plt.title("E3: LDA, shrinkage of the shared covariance")
plt.xlabel("shrinkage (1 = identity, i.e. no LDA)")
plt.ylabel("validation top-1 accuracy")
plt.legend(frameon=False)
plt.savefig(OUT / "fig_ncm_lda.png", dpi=150, bbox_inches="tight")
plt.close()

# E4 learning curve (last point = full training set)
plt.figure(figsize=(6.2, 3.2))
for name, color in zip(models, ("#2a78d6", "#eb6834")):
    part = curve[curve.model == name]
    plt.plot(part.per_class, part.val_top1, "o-", label=name, color=color)
plt.xscale("log")
plt.title("E4: learning curve")
plt.xlabel("training images per class")
plt.ylabel("validation top-1 accuracy")
plt.legend(frameon=False)
plt.savefig(OUT / "fig_ncm_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# final accuracy on the three sets
x = np.arange(3)
plt.figure(figsize=(7, 3.2))
width = 0.8 / len(models)
for j, (name, color) in enumerate(zip(models, ("#2a78d6", "#eb6834"))):
    r = results[results.model == name]
    shift = (j - (len(models) - 1) / 2) * width
    plt.bar(x + shift, r.top1, width, label=name, color=color)
    for i, v in enumerate(r.top1):
        plt.text(i + shift, v + 0.01, "%.1f%%" % (100 * v), ha="center", fontsize=9)
plt.xticks(x, ["our validation set", "test set 1 (ImageNet val)", "test set 2 (ImageNetV2)"], fontsize=9)
plt.ylim(0, 1.15)
plt.title("Nearest centroid trained on the full training set")
plt.ylabel("top-1 accuracy")
plt.legend(frameon=False, loc="upper center", ncol=2, fontsize=9)
plt.savefig(OUT / "fig_ncm_test_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()

# how close is each image to the centroid of its own class? (cosine similarity, chosen model)
# a histogram per set, like the pixel histograms in Lab 8: lower similarity = less typical images
plt.figure(figsize=(6.2, 3.2))
for s, color in (("val", "grey"), ("test1", "#2a78d6"), ("test2", "#eb6834")):
    X, y = sets[s]
    sim = (l2_normalise(best_transform(X)) * l2_normalise(final[best_name])[y]).sum(axis=1)
    histogram, bin_edges = np.histogram(sim, bins=50, range=(-0.2, 1), density=True)
    plt.plot(bin_edges[:-1], histogram, color=color, label="%s (mean %.3f)" % (s, sim.mean()))
plt.title("Similarity to the own-class centroid")
plt.xlabel("cosine similarity")
plt.ylabel("density")
plt.legend(frameon=False)
plt.savefig(OUT / "fig_ncm_centroid_similarity.png", dpi=150, bbox_inches="tight")
plt.close()
