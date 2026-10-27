# 04_linear_classifier.py - classifier A: L2-regularised multinomial logistic regression on EVA-02 features

import time

import pandas as pd
import matplotlib.pyplot as plt
import torch

from common import OUT, SETS, N_CLASSES, DEVICE, BLUE, ORANGE, GREY, training_tensor, standardised, predict, train, \
    accuracy, save_predictions

plt.rcParams.update({"font.size": 11})

WEIGHT_DECAYS = [0.1, 1.0, 10.0, 30.0, 100.0]
TUNE_EPOCHS = 10
FULL_EPOCHS = 10


def linear():
    return torch.nn.Linear(1024, N_CLASSES)


def by_val_top1(row):
    return row["val_top1"]


# 1. Load, standardised with the training split's mean and std
X_train, y_train, mean, std = training_tensor()
X_small, y_small = standardised("train_small", mean, std)
val = standardised("val", mean, std)
print("device:", DEVICE, "| training split", tuple(X_train.shape))

# 2. Tune the weight decay on train_small
tuning = []
for wd in WEIGHT_DECAYS:
    _, history = train(linear, X_small, y_small, TUNE_EPOCHS, by_val_top1, val, wd)
    tuning.append(history.assign(weight_decay=wd))
tuning = pd.concat(tuning, ignore_index=True)
tuning.to_csv(OUT / "linear_tuning.csv", index=False)
wd = tuning.loc[tuning.val_top1.idxmax(), "weight_decay"]
print("chosen weight decay:", wd)

# 3. Train on the full training split, keeping the epoch with the best validation top-1
t0 = time.time()
model, history = train(linear, X_train, y_train, FULL_EPOCHS, by_val_top1, val, wd)
history.to_csv(OUT / "linear_history.csv", index=False)
final = {"weight_decay": wd, "best_epoch": int(history.loc[history.kept, "epoch"].iloc[0]),
         "train_rows": len(y_train), "train_seconds": time.time() - t0, "device": DEVICE}

# 4. Score once on both test sets
for s in SETS:
    X, y = standardised(s, mean, std)
    top5 = predict(model, X)
    save_predictions("linear", s, y, top5)
    final[f"{s}_top1"], final[f"{s}_top5"] = accuracy(top5, y)
pd.DataFrame([final]).to_csv(OUT / "linear_final.csv", index=False)
print(pd.Series(final).to_string())

# 5. Figure: validation top-1 per epoch
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for wd_k, color in zip(WEIGHT_DECAYS, (BLUE, ORANGE, GREY, "k", "tab:green")):
    part = tuning[tuning.weight_decay == wd_k]
    axes[0].plot(part.epoch, part.val_top1, "o-", color=color, label=f"weight decay {wd_k:g}")
axes[0].set_title("Tuning on train_small (100 per class)")
axes[1].plot(history.epoch, history.val_top1, "o-", color=BLUE, label=f"weight decay {wd:g}")
kept = history[history.kept]
axes[1].scatter(kept.epoch, kept.val_top1, s=90, facecolors="none", edgecolors="k", zorder=3, label="kept epoch")
axes[1].set_title("Full training split")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation top-1 accuracy")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig_linear_learning.png", dpi=150, bbox_inches="tight")
plt.close()
