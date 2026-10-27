# 02_prepare_data.py - EVA-02 feature CSVs to .npy, our validation set and train_small

import time
import zipfile

import numpy as np
import pandas as pd

from common import RAW, PROC, OUT, SEED, N_CLASSES, CHUNK

MODEL = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
VAL_PER_CLASS = 50
SMALL_PER_CLASS = 100
PROC.mkdir(parents=True, exist_ok=True)


# 1. Convert: read the CSVs in chunks into float32 arrays on disk (skipped if already done)
def open_raw(name):
    for path in (RAW / f"{name}_{MODEL}.csv", RAW / f"{name}_{MODEL}.csv.zip"):
        if path.exists() and path.suffix == ".csv":
            return open(path, "rb")
        if path.exists() and path.suffix == ".zip":
            z = zipfile.ZipFile(path)
            return z.open([m for m in z.namelist() if m.endswith(".csv")][0])
    raise FileNotFoundError(f"{name}_{MODEL}.csv(.zip) not found in {RAW}")


def convert(name, out):
    if (PROC / f"{out}_X.npy").exists():
        print(out, "already converted")
        return
    t0 = time.time()
    with open_raw(name) as f:
        features = [c for c in f.readline().decode().strip().split(",") if c.isdigit()]
        n = sum(block.count(b"\n") for block in iter(lambda: f.read(2**26), b""))
    print(f"{out}: {n:,} rows x {len(features)} features")
    X = np.lib.format.open_memmap(PROC / f"{out}_X.npy", mode="w+", dtype=np.float32, shape=(n, len(features)))
    y = np.empty(n, dtype=np.int16)
    paths, start = [], 0
    with open_raw(name) as f:
        dtypes = {c: np.float32 for c in features} | {"label": np.int16}
        for chunk in pd.read_csv(f, usecols=["path", "label"] + features, dtype=dtypes, chunksize=CHUNK):
            X[start:start + len(chunk)] = chunk[features].to_numpy()
            y[start:start + len(chunk)] = chunk["label"].to_numpy()
            paths += chunk["path"].tolist()
            start += len(chunk)
    X.flush()
    del X
    np.save(PROC / f"{out}_y.npy", y)
    pd.DataFrame({"path": paths, "label": y}).to_csv(PROC / f"{out}_meta.csv", index=False)
    print(f"{out} done in {time.time() - t0:.0f}s")


convert("v2", "test2")
convert("val", "test1")
convert("train", "train_full")


# 2. Split: per class, 50 training images to our validation set, 100 of the rest to train_small
y = np.load(PROC / "train_full_y.npy")
rng = np.random.default_rng(SEED)
val_idx, train_idx, small_idx = [], [], []
for c in range(N_CLASSES):
    rows = rng.permutation(np.flatnonzero(y == c))
    val_idx.append(rows[:VAL_PER_CLASS])
    train_idx.append(rows[VAL_PER_CLASS:])
    small_idx.append(rows[VAL_PER_CLASS:VAL_PER_CLASS + SMALL_PER_CLASS])
splits = {"val": val_idx, "train": train_idx, "train_small": small_idx}
splits = {k: np.sort(np.concatenate(v)) for k, v in splits.items()}
for k, v in splits.items():
    np.save(PROC / f"split_{k}_idx.npy", v)
print({k: len(v) for k, v in splits.items()})
X_full = np.load(PROC / "train_full_X.npy", mmap_mode="r")
meta = pd.read_csv(PROC / "train_full_meta.csv")
for k in ("val", "train_small"):
    np.save(PROC / f"{k}_X.npy", np.asarray(X_full[splits[k]]))
    np.save(PROC / f"{k}_y.npy", y[splits[k]])
    meta.iloc[splits[k]].to_csv(PROC / f"{k}_meta.csv", index=False)


# 3. Summary: size and class balance per set
summary = []
for k in ("train_full", "train_small", "val", "test1", "test2"):
    X = np.load(PROC / f"{k}_X.npy", mmap_mode="r")
    counts = np.bincount(np.load(PROC / f"{k}_y.npy"), minlength=N_CLASSES)
    summary.append({"set": k, "rows": len(X), "features": X.shape[1], "classes": int((counts > 0).sum()),
                    "min_per_class": int(counts.min()), "max_per_class": int(counts.max())})
summary = pd.DataFrame(summary)
summary.to_csv(OUT / "prepare_summary.csv", index=False)
print(summary.to_string(index=False))
