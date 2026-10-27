# 03_integrity_audit.py - checks the prepared sets and finds exact feature duplicates within and across sets

import hashlib

import numpy as np
import pandas as pd

from common import PROC, OUT, N_CLASSES, CHUNK, load

checks = []


def check(name, value, expected=None):
    ok = "" if expected is None else bool(value == expected)
    checks.append({"check": name, "value": value, "expected": "" if expected is None else expected, "pass": ok})
    print(f"{name}: {value}" + ("" if expected is None else f"  (expected {expected}) {'OK' if ok else 'FAIL'}"))


# 1. Splits
X_full, y_full = load("train_full", mmap=True)
idx = {k: np.load(PROC / f"split_{k}_idx.npy") for k in ("train", "val", "train_small")}
check("training split and validation set share rows", len(np.intersect1d(idx["train"], idx["val"])), 0)
check("training split + validation set cover train_full", len(np.union1d(idx["train"], idx["val"])), len(y_full))
check("train_small rows outside the training split", int((~np.isin(idx["train_small"], idx["train"])).sum()), 0)
for k in ("val", "train_small"):
    Xk, yk = load(k, mmap=True)
    check(f"{k}: labels equal train_full at its rows", bool(np.array_equal(yk, y_full[idx[k]])), True)
    same = all(np.array_equal(np.asarray(Xk[i:i + CHUNK]), np.asarray(X_full[idx[k][i:i + CHUNK]]))
               for i in range(0, len(yk), CHUNK))
    check(f"{k}: features equal train_full at its rows", same, True)


# 2. Shape, labels and finite values per set
sets = {"train_full": (X_full, y_full)} | {k: load(k, mmap=True) for k in ("val", "train_small", "test1", "test2")}
for k, (X, y) in sets.items():
    check(f"{k}: feature dimension", X.shape[1], 1024)
    check(f"{k}: labels within 0-999", bool(y.min() >= 0 and y.max() < N_CLASSES), True)
    check(f"{k}: classes present", len(np.unique(y)), N_CLASSES)
    check(f"{k}: non-finite values",
          int(sum((~np.isfinite(np.asarray(X[i:i + CHUNK]))).sum() for i in range(0, len(X), CHUNK))), 0)


# 3. Exact duplicates: rows with bit-identical features, found by hashing
def digests(X):
    return [hashlib.md5(r.tobytes()).hexdigest() for i in range(0, len(X), CHUNK) for r in np.asarray(X[i:i + CHUNK])]


print("hashing feature rows ...", flush=True)
h_full = digests(X_full)
H = {"training split": [h_full[i] for i in idx["train"]], "validation set": [h_full[i] for i in idx["val"]],
     "test1": digests(sets["test1"][0]), "test2": digests(sets["test2"][0])}
Y = {"training split": y_full[idx["train"]], "validation set": y_full[idx["val"]],
     "test1": sets["test1"][1], "test2": sets["test2"][1]}
labels_of = {k: {} for k in H}                       # digest -> labels it carries in that set
for k in H:
    for d, lab in zip(H[k], Y[k].tolist()):
        labels_of[k].setdefault(d, set()).add(lab)

for k in H:
    _, counts = np.unique(H[k], return_counts=True)
    check(f"{k}: rows whose features repeat within the set", int(counts[counts > 1].sum()))
counts = pd.Series(H["training split"]).value_counts()
repeated = set(counts[counts > 1].index)
check("training split: repeated feature vectors carrying more than one label",
      sum(len(labels_of["training split"][d]) > 1 for d in repeated))
check("training split: repeated feature vectors carrying one label",
      sum(len(labels_of["training split"][d]) == 1 for d in repeated))

names = list(H)
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        ka, kb = names[a], names[b]
        hit = [d in labels_of[ka] for d in H[kb]]
        check(f"{kb} rows matching a {ka} row", int(sum(hit)))
        if any(hit):
            same = sum(lab in labels_of[ka][d] for d, lab, m in zip(H[kb], Y[kb].tolist(), hit) if m)
            check(f"  of which share a label with a matching {ka} row", int(same))

# test set 1 rows that are also training images; later scripts read this file
dups = pd.DataFrame({"in_training_split": [d in labels_of["training split"] for d in H["test1"]],
                     "in_validation_set": [d in labels_of["validation set"] for d in H["test1"]]})
dups["duplicate"] = dups.in_training_split | dups.in_validation_set
dups["label_differs_from_every_copy"] = [
    bool(m) and lab not in labels_of["training split"].get(d, set()) | labels_of["validation set"].get(d, set())
    for d, lab, m in zip(H["test1"], Y["test1"].tolist(), dups.duplicate)]
dups.to_csv(OUT / "audit_test1_duplicates.csv", index_label="row")
check("test1 rows that duplicate a training image", int(dups.duplicate.sum()))
check("  of which with a label different from every copy", int(dups.label_differs_from_every_copy.sum()))

pd.DataFrame(checks).to_csv(OUT / "audit_checks.csv", index=False)
failed = [c["check"] for c in checks if c["pass"] is False]
print("\nall checks with an expected value passed" if not failed else f"\nFAILED: {failed}")
