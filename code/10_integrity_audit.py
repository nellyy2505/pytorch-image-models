# 10_integrity_audit.py - checks the prepared splits at record and content level.
# Record level: split indices, array consistency, labels, dimensions and finite values. Content level: exact
# feature-vector matches within and across splits, and whether matched rows carry different labels.
# Reads prepared data only; run after 02_prepare_data.py (hashing the training array takes a few minutes).

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC, OUT = ROOT / "data" / "processed", ROOT / "data" / "output"
CHUNK = 100_000
checks = []


def check(name, value, expected=None):
    ok = "" if expected is None else bool(value == expected)
    checks.append({"check": name, "value": value, "expected": "" if expected is None else expected, "pass": ok})
    print(f"{name}: {value}" + ("" if expected is None else f"  (expected {expected}) {'OK' if ok else 'FAIL'}"))


# 1. Record-level split checks
X_full = np.load(PROC / "train_full_X.npy", mmap_mode="r")
y_full = np.load(PROC / "train_full_y.npy").astype(int)
idx = {k: np.load(PROC / f"split_{k}_idx.npy") for k in ("train", "val", "train_small")}
check("rows in train_full", len(y_full))
check("training split and validation set share indices", len(np.intersect1d(idx["train"], idx["val"])), 0)
check("training split + validation set cover train_full", len(np.union1d(idx["train"], idx["val"])), len(y_full))
check("train_small indices outside the training split", int((~np.isin(idx["train_small"], idx["train"])).sum()), 0)
check("train_small indices inside the validation set", len(np.intersect1d(idx["train_small"], idx["val"])), 0)
for k in ("val", "train_small"):
    Xk, yk = np.load(PROC / f"{k}_X.npy", mmap_mode="r"), np.load(PROC / f"{k}_y.npy").astype(int)
    check(f"{k}: labels equal train_full labels at its indices", bool(np.array_equal(yk, y_full[idx[k]])), True)
    same = all(np.array_equal(np.asarray(Xk[i:i + CHUNK]), np.asarray(X_full[idx[k][i:i + CHUNK]]))
               for i in range(0, len(yk), CHUNK))
    check(f"{k}: features equal train_full features at its indices", same, True)


# 2. Shape, labels and finite values per set
sets = {"train_full": (X_full, y_full)}
for k in ("val", "train_small", "test1", "test2"):
    sets[k] = (np.load(PROC / f"{k}_X.npy", mmap_mode="r"), np.load(PROC / f"{k}_y.npy").astype(int))
for k, (X, y) in sets.items():
    check(f"{k}: feature dimension", X.shape[1], 1024)
    check(f"{k}: labels within 0-999", bool(y.min() >= 0 and y.max() <= 999), True)
    check(f"{k}: classes present", len(np.unique(y)), 1000)
    check(f"{k}: non-finite values", int(sum((~np.isfinite(np.asarray(X[i:i + CHUNK]))).sum()
                                             for i in range(0, len(X), CHUNK))), 0)


# 3. Content-level checks: exact feature-vector matches
def hashes(X):
    return np.array([hashlib.md5(r.tobytes()).digest() for i in range(0, len(X), CHUNK)
                     for r in np.asarray(X[i:i + CHUNK])], dtype="S16")


print("hashing feature rows ...", flush=True)
h_full = hashes(X_full)
H = {"training split": h_full[idx["train"]], "validation set": h_full[idx["val"]],
     "test1": hashes(sets["test1"][0]), "test2": hashes(sets["test2"][0])}
Y = {"training split": y_full[idx["train"]], "validation set": y_full[idx["val"]],
     "test1": sets["test1"][1], "test2": sets["test2"][1]}

for k, h in H.items():
    _, counts = np.unique(h, return_counts=True)
    check(f"{k}: rows whose features repeat within the set", int(counts[counts > 1].sum()))

names = list(H)
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        ka, kb = names[a], names[b]
        rows_b = np.isin(H[kb], H[ka])
        check(f"{kb} rows whose features match a {ka} row", int(rows_b.sum()))
        if rows_b.any():
            label_of = dict(zip(H[ka].tolist(), Y[ka].tolist()))      # one label per hash (first is enough)
            conflict = sum(label_of[h] != lab for h, lab in zip(H[kb][rows_b].tolist(), Y[kb][rows_b].tolist()))
            check(f"  of which with a different label", int(conflict))

pd.DataFrame(checks).to_csv(OUT / "integrity_audit.csv", index=False)
failed = [c["check"] for c in checks if c["pass"] is False]
print("\nall record-level checks passed" if not failed else f"\nFAILED: {failed}")
