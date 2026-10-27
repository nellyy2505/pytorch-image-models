# 01_practice_classifier.py - practice task: logistic regression on EfficientFormerV2-S0 features (176-d)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RAW, OUT, SEED, N_CLASSES, BLUE, ORANGE

plt.rcParams.update({"font.size": 11})

MODEL = "efficientformerv2_s0.snap_dist_in1k"
PER_CLASS = 200           # all 1.28 M rows do not fit lbfgs in memory
MAX_ITER = 300


# 1. Load
def load(name):
    path = RAW / f"{name}_{MODEL}.csv"
    features = [c for c in pd.read_csv(path, nrows=0).columns if c.isdigit()]
    df = pd.read_csv(path, usecols=["label"] + features, dtype={c: np.float32 for c in features} | {"label": np.int16})
    return df[features].to_numpy(), df["label"].to_numpy()


X_train_all, y_train_all = load("train")
X_test1, y_test1 = load("val")
X_test2, y_test2 = load("v2")
print("train:", X_train_all.shape, "test set 1:", X_test1.shape, "test set 2:", X_test2.shape)


# 2. Train on a class-stratified sample
rng = np.random.default_rng(SEED)
idx = np.concatenate([rng.choice(np.flatnonzero(y_train_all == c), PER_CLASS, replace=False) for c in range(N_CLASSES)])
X_train, y_train = X_train_all[idx], y_train_all[idx]
del X_train_all
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=MAX_ITER, random_state=SEED))
model.fit(X_train, y_train)


# 3. Evaluate on both test sets
def top_k(X, y, k):
    top = model.classes_[np.argsort(-model.predict_proba(X), axis=1)[:, :k]]
    return (top == y[:, None]).any(axis=1).mean()


results = pd.DataFrame({
    "test_set": ["test set 1 (ImageNet val)", "test set 2 (ImageNetV2)"],
    "rows": [len(y_test1), len(y_test2)],
    "top1": [top_k(X_test1, y_test1, 1), top_k(X_test2, y_test2, 1)],
    "top5": [top_k(X_test1, y_test1, 5), top_k(X_test2, y_test2, 5)],
})
results.to_csv(OUT / "practice_results.csv", index=False)
print(results.round(4).to_string(index=False))
print("top-1 gap: %.1f points" % (100 * (results.top1[0] - results.top1[1])))

fig, ax = plt.subplots(figsize=(6.2, 3))
x = np.arange(2)
ax.bar(x - 0.2, results.top1, 0.4, label="top-1", color=BLUE)
ax.bar(x + 0.2, results.top5, 0.4, label="top-5", color=ORANGE)
for i in range(2):
    ax.text(i - 0.2, results.top1[i] + 0.01, "%.1f%%" % (100 * results.top1[i]), ha="center")
    ax.text(i + 0.2, results.top5[i] + 0.01, "%.1f%%" % (100 * results.top5[i]), ha="center")
ax.set_xticks(x, results.test_set)
ax.set_ylim(0, 1.15)
ax.set_ylabel("accuracy")
ax.set_title("Logistic regression on EfficientFormerV2-S0 features")
ax.legend(frameon=False, loc="upper center", ncol=2)
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_practice_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
