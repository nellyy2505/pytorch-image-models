# common.py - paths, constants and helpers shared by the numbered scripts (imported, not run)

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 7
N_CLASSES = 1000
CHUNK = 20_000
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#888888"
SETS = {"val": "our validation set", "test1": "test set 1 (ImageNet val)", "test2": "test set 2 (ImageNetV2)"}
MODELS = {"linear": "Regularised linear (A)", "mlp": "Small MLP (B)", "centroid": "Nearest centroid (C)",
          "refined": "Refined linear (A + R1/R2)"}


# 1. Data
def load(name, mmap=False):
    X = np.load(PROC / f"{name}_X.npy", mmap_mode="r" if mmap else None)
    return X, np.load(PROC / f"{name}_y.npy").astype(np.int64)


def training_split():
    # memory-mapped training array, labels, and the rows of our training split (all but our validation set)
    X, y = load("train_full", mmap=True)
    return X, y, np.load(PROC / "split_train_idx.npy")


def class_sums(Z, y):
    onehot = sparse.csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))), shape=(N_CLASSES, len(y)))
    return np.asarray(onehot @ np.asarray(Z, dtype=np.float64))


def unit(Z):
    Z = np.asarray(Z, dtype=np.float64)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


# 2. Typicality: cosine similarity to the own-class centroid of train_small (label-free: to the closest centroid)
_centroids = None


def centroids():
    global _centroids
    if _centroids is None:
        X, y = load("train_small")
        _centroids = unit(class_sums(unit(X), y))
    return _centroids


def typicality(X, y):
    C = centroids()
    return np.concatenate([(unit(X[i:i + CHUNK]) * C[y[i:i + CHUNK]]).sum(axis=1) for i in range(0, len(X), CHUNK)])


def typicality_label_free(X):
    C = centroids()
    return np.concatenate([(unit(X[i:i + CHUNK]) @ C.T).max(axis=1) for i in range(0, len(X), CHUNK)])


# 3. Predictions: the 5 best classes per image, in the row order of <set>_X.npy
def accuracy(top5, y):
    return (top5[:, 0] == y).mean(), (top5 == y[:, None]).any(axis=1).mean()


def save_predictions(model, s, y, top5):
    pred = pd.DataFrame(top5, columns=[f"pred{k}" for k in range(1, 6)])
    pred.insert(0, "label", y)
    pred.to_csv(OUT / f"pred_{model}_{s}.csv", index_label="row")


def read_predictions(model, s):
    df = pd.read_csv(OUT / f"pred_{model}_{s}.csv")
    y = np.load(PROC / f"{s}_y.npy").astype(np.int64)
    if not (np.array_equal(df.row, np.arange(len(y))) and np.array_equal(df.label, y)):
        raise ValueError(f"pred_{model}_{s}.csv is not in the row order of {s}_y.npy; re-run its script")
    return df[[f"pred{k}" for k in range(1, 6)]].to_numpy()


# 4. Neural classifiers (A linear, B MLP, refinement) share one training loop; GPU if there is one
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")             # for deterministic cuBLAS
torch.use_deterministic_algorithms(True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1024


def training_tensor():
    # the training split standardised with its own mean and std, kept in RAM (5 GB); batches go to the GPU
    X_full, y_full, rows = training_split()
    X = torch.empty((len(rows), X_full.shape[1]), dtype=torch.float32)
    for i in range(0, len(rows), CHUNK):
        X[i:i + CHUNK] = torch.from_numpy(np.asarray(X_full[rows[i:i + CHUNK]]))
    mean, std = X.mean(dim=0), X.std(dim=0) + 1e-6
    for i in range(0, len(X), CHUNK):
        X[i:i + CHUNK] = (X[i:i + CHUNK] - mean) / std
    return X, y_full[rows], mean, std


def standardised(name, mean, std):
    X, y = load(name)
    return (torch.from_numpy(X) - mean) / std, y


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + 8192].to(DEVICE)).topk(5, dim=1).indices.cpu()
                          for i in range(0, len(X), 8192)]).numpy()


def train(make_model, X, y, epochs, choose, val, weight_decay, weights=None, spread=0.0, val_masks=None):
    # AdamW with the learning rate decayed to 0 over all epochs (cosine); keeps the epoch maximising choose(row),
    # or the best validation top-1 epoch if choose rejects all; weights (R1) and spread (R2) are for 08
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    model = make_model().to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs * -(-len(y) // BATCH))
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    y_t = torch.from_numpy(y)
    w_t = torch.ones(len(y)) if weights is None else torch.as_tensor(weights, dtype=torch.float32)
    if spread > 0:
        means = torch.zeros((N_CLASSES, X.shape[1]))
        for i in range(0, len(X), CHUNK):
            means += torch.from_numpy(class_sums(X[i:i + CHUNK].numpy(), y[i:i + CHUNK])).float()
        means = (means / torch.from_numpy(np.bincount(y, minlength=N_CLASSES)).float()[:, None]).to(DEVICE)
    history, states = [], {}
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        order = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), BATCH):
            batch = order[i:i + BATCH]
            Z, target = X[batch].to(DEVICE), y_t[batch].to(DEVICE)
            if spread > 0:
                c = means[target]
                Z = c + (1 + spread * torch.rand(len(batch), 1, generator=g)).to(DEVICE) * (Z - c)
            loss = (loss_fn(model(Z), target) * w_t[batch].to(DEVICE)).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
        top5 = predict(model, val[0])
        row = dict(zip(("val_top1", "val_top5"), accuracy(top5, val[1])), epoch=epoch, seconds=time.time() - t0)
        for name, mask in (val_masks or {}).items():
            row[f"val_{name}_top1"] = (top5[mask, 0] == val[1][mask]).mean()
        history.append(row)
        print({k: round(float(v), 5) for k, v in row.items()}, flush=True)
        states[epoch] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    history = pd.DataFrame(history)
    history["score"] = [choose(r) for r in history.to_dict("records")]
    history["meets_rule"] = np.isfinite(history.score)
    best = history.score.idxmax() if history.meets_rule.any() else history.val_top1.idxmax()
    history["kept"] = history.index == best
    model.load_state_dict(states[int(history.epoch[best])])
    return model, history
