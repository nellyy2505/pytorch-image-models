# 05_mlp_classifier.py - classifier B: a one-hidden-layer MLP on EVA-02 features

import itertools
import time

import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from common import OUT, SETS, N_CLASSES, DEVICE, BLUE, ORANGE, training_tensor, standardised, predict, train, \
    accuracy, save_predictions

plt.rcParams.update({"font.size": 11})

TUNE_EPOCHS = 10
FINAL_EPOCHS = 20
GRID = {"hidden": [512, 1024, 2048], "dropout": [0.0, 0.5, 0.7], "weight_decay": [0.1, 1.0, 10.0]}


def mlp(hidden, dropout):
    return lambda: nn.Sequential(nn.Linear(1024, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, N_CLASSES))


def by_val_top1(row):
    return row["val_top1"]


# 1. Load, standardised with the training split's mean and std
X_train, y_train, mean, std = training_tensor()
X_small, y_small = standardised("train_small", mean, std)
val = standardised("val", mean, std)
print("device:", DEVICE, "| training split", tuple(X_train.shape))

# 2. Tune on train_small
tuning = []
for hidden, dropout, wd in itertools.product(*GRID.values()):
    t0 = time.time()
    _, history = train(mlp(hidden, dropout), X_small, y_small, TUNE_EPOCHS, by_val_top1, val, wd)
    best = history.loc[history.val_top1.idxmax()]
    tuning.append({"hidden": hidden, "dropout": dropout, "weight_decay": wd, "best_epoch": int(best.epoch),
                   "val_top1": best.val_top1, "val_top5": best.val_top5, "seconds": time.time() - t0})
tuning = pd.DataFrame(tuning)
tuning.to_csv(OUT / "mlp_tuning.csv", index=False)
chosen = tuning.loc[tuning.val_top1.idxmax()]
print(tuning.round(4).to_string(index=False))

# 3. Chosen setting on the full training split, keeping the epoch with the best validation top-1
t0 = time.time()
model, history = train(mlp(int(chosen.hidden), chosen.dropout), X_train, y_train, FINAL_EPOCHS, by_val_top1, val,
                       chosen.weight_decay)
history.to_csv(OUT / "mlp_history.csv", index=False)
final = {"hidden": int(chosen.hidden), "dropout": chosen.dropout, "weight_decay": chosen.weight_decay,
         "best_epoch": int(history.loc[history.kept, "epoch"].iloc[0]), "train_rows": len(y_train),
         "train_seconds": time.time() - t0, "device": DEVICE}

# 4. Score once on both test sets
for s in SETS:
    X, y = standardised(s, mean, std)
    top5 = predict(model, X)
    save_predictions("mlp", s, y, top5)
    final[f"{s}_top1"], final[f"{s}_top5"] = accuracy(top5, y)
pd.DataFrame([final]).to_csv(OUT / "mlp_final.csv", index=False)
print(pd.Series(final).to_string())

fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(history.epoch, history.val_top1, "o-", color=BLUE, label="validation top-1")
ax.plot(history.epoch, history.val_top5, "o-", color=ORANGE, label="validation top-5")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
ax.set_title("MLP on the full training split: our validation set")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_mlp_validation.png", dpi=150, bbox_inches="tight")
plt.close()
