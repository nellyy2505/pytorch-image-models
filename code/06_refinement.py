# 07_refinement.py - part (c): test the hypothesis from the generalisation analysis by refining the classifier
#
# Hypothesis (section 5): test set 2 (ImageNetV2) holds a larger share of atypical images, i.e. images far from
# their own class centre in EVA-02 feature space, and at equal typicality both test sets are equally accurate.
# If that is right, a classifier that does better on atypical images should narrow the gap.
#
# Two refinements of a linear softmax classifier (the same model family as classifier A, written in PyTorch so
# every variant trains with the same code and in minutes), each a direct consequence of the hypothesis:
#   R1 weighting:  training images get a weight that grows with how atypical they are within their class
#   R2 spreading:  each training image is pushed away from its class centre by a random factor, which widens
#                  every class the way test set 2's classes are wider
# Typicality is the cosine similarity of an image to its own class centroid from train_small, the same measure
# section 5 uses. Training images only need their own label, which is known, so nothing touches the test sets.
#
# Selection: every setting is trained on train_small and scored on our validation set. The setting with the best
# top-1 on the least typical 10% of our validation set wins, provided its overall validation top-1 stays within
# 0.2 points of the unweighted baseline. Then the baseline and the winner are trained on the full training set,
# and both are scored once on test set 1 and test set 2.
#
# Run from the repository root:  python code/07_refinement.py          (about 20-40 minutes on a laptop CPU)

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 7
N_CLASSES = 1000
BATCH = 1024
LR, WEIGHT_DECAY = 1e-3, 1e-4
SMALL_EPOCHS, FULL_EPOCHS = 15, 6
ATYPICAL_SHARE = 0.10           # the least typical 10% of our validation set is the yardstick
MAX_VAL_DROP = 0.002            # a refinement may cost at most 0.2 points of overall validation top-1
SETTINGS = ([("baseline", 0.0, 0.0)]
            + [("R1 weighting", beta, 0.0) for beta in (1.0, 2.0, 4.0)]
            + [("R2 spreading", 0.0, gamma) for gamma in (0.25, 0.5, 1.0)])
CHUNK = 100_000

torch.manual_seed(SEED)
torch.set_num_threads(os.cpu_count() or 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")


# 1. Load
def load(name):
    return np.load(PROC / f"{name}_X.npy"), np.load(PROC / f"{name}_y.npy").astype(np.int64)


X_small, y_small = load("train_small")
X_val, y_val = load("val")
X_test = {t: load(t) for t in ("test1", "test2")}

# the full training rows (all except our validation set) are kept in memory as float16: 2.5 GB instead of 5 GB
X_full_map = np.load(PROC / "train_full_X.npy", mmap_mode="r")
y_full_all = np.load(PROC / "train_full_y.npy").astype(np.int64)
train_idx = np.load(PROC / "split_train_idx.npy")
X_full = np.empty((len(train_idx), X_full_map.shape[1]), dtype=np.float16)
for i in range(0, len(train_idx), CHUNK):
    X_full[i:i + CHUNK] = X_full_map[train_idx[i:i + CHUNK]]
y_full = y_full_all[train_idx]
print("train_small", X_small.shape, "train_full", X_full.shape, "val", X_val.shape, "device", DEVICE)


# 2. Statistics from training data only
def unit(Z):
    Z = np.asarray(Z, dtype=np.float32)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


# typicality: cosine similarity to the own-class centroid of train_small (as in section 5)
centroid_raw = np.zeros((N_CLASSES, X_small.shape[1]))
np.add.at(centroid_raw, y_small, unit(X_small))
centroid_raw = unit(centroid_raw)


def typicality(X, y):
    out = np.empty(len(y), dtype=np.float32)
    for i in range(0, len(y), CHUNK):
        out[i:i + CHUNK] = (unit(X[i:i + CHUNK]) * centroid_raw[y[i:i + CHUNK]]).sum(axis=1)
    return out


def standardiser(X):
    s, s2, n = 0.0, 0.0, 0
    for i in range(0, len(X), CHUNK):
        Z = np.asarray(X[i:i + CHUNK], dtype=np.float64)
        s, s2, n = s + Z.sum(axis=0), s2 + (Z ** 2).sum(axis=0), n + len(Z)
    mean = s / n
    return mean.astype(np.float32), np.sqrt(s2 / n - mean ** 2).astype(np.float32) + 1e-6


def class_means(X, y, mean, std):
    sums = np.zeros((N_CLASSES, X.shape[1]))
    for i in range(0, len(y), CHUNK):
        np.add.at(sums, y[i:i + CHUNK], (np.asarray(X[i:i + CHUNK], dtype=np.float32) - mean) / std)
    return (sums / np.bincount(y, minlength=N_CLASSES)[:, None]).astype(np.float32)


def atypicality_weights(X, y, beta):
    # rank of each image's atypicality within its class: 0 = most typical, 1 = least typical
    t = typicality(X, y)
    order = np.lexsort((t, y))                       # by class, then by typicality ascending
    rank = np.empty(len(y), dtype=np.float32)
    starts = np.r_[0, np.cumsum(np.bincount(y, minlength=N_CLASSES))]
    for c in range(N_CLASSES):
        rows = order[starts[c]:starts[c + 1]]
        rank[rows] = 1 - np.arange(len(rows)) / max(len(rows) - 1, 1)
    w = 1 + beta * rank
    return w / w.mean()


# the least typical 10% of our validation set
t_val = typicality(X_val, y_val)
val_atypical = t_val <= np.quantile(t_val, ATYPICAL_SHARE)


# 3. Train and score
def predict_top5(model, X, mean, std):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            Z = torch.from_numpy((np.asarray(X[i:i + 8192], dtype=np.float32) - mean) / std).to(DEVICE)
            out.append(model(Z).topk(5, dim=1).indices.cpu().numpy())
    return np.concatenate(out)


def scores(top5, y, mask=None):
    if mask is not None:
        top5, y = top5[mask], y[mask]
    return (top5[:, 0] == y).mean(), (top5 == y[:, None]).any(axis=1).mean()


def train(X, y, beta, gamma, epochs):
    # linear softmax classifier; keeps the epoch with the best overall validation top-1
    mean, std = standardiser(X)
    weights = torch.from_numpy(atypicality_weights(X, y, beta)) if beta > 0 else torch.ones(len(y))
    centres = torch.from_numpy(class_means(X, y, mean, std)).to(DEVICE) if gamma > 0 else None
    g = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    model = torch.nn.Linear(X.shape[1], N_CLASSES).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    mean_t, std_t = torch.from_numpy(mean).to(DEVICE), torch.from_numpy(std).to(DEVICE)
    best, best_state, history = -1, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y), generator=g).numpy()
        for i in range(0, len(y), BATCH):
            rows = np.sort(order[i:i + BATCH])
            Z = (torch.from_numpy(np.asarray(X[rows], dtype=np.float32)).to(DEVICE) - mean_t) / std_t
            target = torch.from_numpy(y[rows]).to(DEVICE)
            if centres is not None:
                # R2: move each image away from its class centre by a factor between 1 and 1 + gamma
                spread = 1 + gamma * torch.rand(len(rows), 1, generator=g).to(DEVICE)
                Z = centres[target] + spread * (Z - centres[target])
            loss = (loss_fn(model(Z), target) * weights[rows].to(DEVICE)).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        top5 = predict_top5(model, X_val, mean, std)
        val1, _ = scores(top5, y_val)
        atyp1, _ = scores(top5, y_val, val_atypical)
        history.append({"epoch": epoch, "val_top1": val1, "val_atypical_top1": atyp1})
        if val1 > best:
            best, best_state = val1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, mean, std, pd.DataFrame(history)


# 4. Choose the refinement on train_small and our validation set
grid = []
for name, beta, gamma in SETTINGS:
    t0 = time.time()
    model, mean, std, history = train(X_small, y_small, beta, gamma, SMALL_EPOCHS)
    top5 = predict_top5(model, X_val, mean, std)
    val1, val5 = scores(top5, y_val)
    atyp1, _ = scores(top5, y_val, val_atypical)
    grid.append({"refinement": name, "beta": beta, "gamma": gamma, "best_epoch": int(history.val_top1.idxmax()) + 1,
                 "val_top1": val1, "val_top5": val5, "val_atypical_top1": atyp1, "seconds": time.time() - t0})
    print(grid[-1])
grid = pd.DataFrame(grid)
base = grid.iloc[0]
allowed = grid[(grid.index > 0) & (grid.val_top1 >= base.val_top1 - MAX_VAL_DROP)]
chosen = allowed.loc[allowed.val_atypical_top1.idxmax()] if len(allowed) else base
grid["chosen"] = grid.index == chosen.name
grid.to_csv(OUT / "refine_grid.csv", index=False)
print(grid.round(4).to_string(index=False))
print("chosen:", chosen.refinement, "beta", chosen.beta, "gamma", chosen.gamma)


# 5. Train baseline and chosen refinement on the full training set; score once on both test sets
results, predictions, typ_rows = [], [], []
t_test = {t: typicality(*X_test[t]) for t in X_test}
edges = np.quantile(t_test["test1"], np.linspace(0, 1, 11))[1:-1]
runs = [("baseline", 0.0, 0.0), (f"{chosen.refinement} (beta={chosen.beta:g}, gamma={chosen.gamma:g})",
                                  chosen.beta, chosen.gamma)]
for name, beta, gamma in runs:
    t0 = time.time()
    model, mean, std, history = train(X_full, y_full, beta, gamma, FULL_EPOCHS)
    train_s = time.time() - t0
    history.assign(run=name).to_csv(OUT / f"refine_history_{'baseline' if beta == gamma == 0 else 'refined'}.csv",
                                    index=False)
    row = {"run": name, "beta": beta, "gamma": gamma, "best_epoch": int(history.val_top1.idxmax()) + 1,
           "train_seconds": train_s}
    top5 = predict_top5(model, X_val, mean, std)
    row["val_top1"], row["val_top5"] = scores(top5, y_val)
    row["val_atypical_top1"], _ = scores(top5, y_val, val_atypical)
    for t, (X, y) in X_test.items():
        top5 = predict_top5(model, X, mean, std)
        row[f"{t}_top1"], row[f"{t}_top5"] = scores(top5, y)
        bins = np.digitize(t_test[t], edges)
        for b in range(10):
            typ_rows.append({"run": name, "test_set": t, "decile": b + 1, "share": (bins == b).mean(),
                             "top1": (top5[bins == b, 0] == y[bins == b]).mean()})
        predictions.append(pd.DataFrame({"run": name, "test_set": t, "row": np.arange(len(y)), "label": y,
                                         **{f"top{k + 1}": top5[:, k] for k in range(5)}}))
    row["gap_top1"] = row["test1_top1"] - row["test2_top1"]
    results.append(row)
    print(row)

results = pd.DataFrame(results)
results.to_csv(OUT / "refine_results.csv", index=False)
pd.DataFrame(typ_rows).to_csv(OUT / "refine_typicality.csv", index=False)
pd.concat(predictions).to_csv(OUT / "refine_predictions.csv", index=False)
print(results.round(4).to_string(index=False))
print("gap change (refined - baseline): %+.2f points" % (100 * (results.gap_top1[1] - results.gap_top1[0])))


# 6. Figures: before/after on both test sets, and test set 2 accuracy by typicality decile
BLUE, ORANGE = "#2a78d6", "#eb6834"
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(2)
for k, (t, label, color) in enumerate((("test1", "test set 1 (ImageNet val)", BLUE),
                                       ("test2", "test set 2 (ImageNetV2)", ORANGE))):
    vals = results[f"{t}_top1"].to_numpy()
    axes[0].bar(x + (k - 0.5) * 0.4, vals, 0.4, color=color, label=label)
    for i, v in enumerate(vals):
        axes[0].text(i + (k - 0.5) * 0.4, v + 0.003, "%.1f" % (100 * v), ha="center", fontsize=9)
axes[0].set_xticks(x)
axes[0].set_xticklabels(["baseline", "refined"])
top1 = results[["test1_top1", "test2_top1"]].to_numpy()
axes[0].set_ylim(top1.min() - 0.06, min(1.0, top1.max() + 0.03))
axes[0].set_ylabel("top-1 accuracy")
axes[0].set_title("Before and after the refinement")
axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
typ = pd.DataFrame(typ_rows)
for (run, part), style in zip(typ[typ.test_set == "test2"].groupby("run", sort=False), ("o--", "o-")):
    axes[1].plot(part.decile, part.top1, style, color=ORANGE, label=run if len(run) < 30 else "refined")
axes[1].set_xlabel("typicality decile (1 = least typical)")
axes[1].set_ylabel("top-1 accuracy on test set 2")
axes[1].set_title("Test set 2 by typicality")
axes[1].legend(frameon=False, fontsize=9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig_refinement.png", dpi=150, bbox_inches="tight")
plt.close()
