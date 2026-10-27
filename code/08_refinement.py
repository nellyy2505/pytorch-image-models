# 08_refinement.py - refine classifier A for atypical images: R1 up-weights them, R2 widens every class
# kept epoch: best top-1 on the least typical 10% of our validation set, within 0.2 points of A's overall top-1

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from common import OUT, SEED, N_CLASSES, CHUNK, SETS, MODELS, DEVICE, BLUE, ORANGE, load, training_split,     training_tensor, standardised, predict, train, accuracy, typicality, save_predictions, read_predictions

plt.rcParams.update({"font.size": 11})

SETTINGS = ([("R1 weighting", beta, 0.0) for beta in (1.0, 2.0, 4.0, 8.0, 16.0)]
            + [("R2 spreading", 0.0, gamma) for gamma in (0.25, 0.5, 1.0)])
ATYPICAL_SHARE = 0.10
MAX_VAL_DROP = 0.002
TUNE_EPOCHS = 10
FULL_EPOCHS = 10
DRAWS = 2000
rng = np.random.default_rng(SEED)


def linear():
    return torch.nn.Linear(1024, N_CLASSES)


# 1. Data, A's settings and the yardstick (least typical 10% of our validation set)
X_train, y_train, mean, std = training_tensor()
X_small, y_small = standardised("train_small", mean, std)
val = standardised("val", mean, std)
a = pd.read_csv(OUT / "linear_final.csv").iloc[0]
t_val = typicality(*load("val"))
atypical = t_val <= np.quantile(t_val, ATYPICAL_SHARE)
print("device:", DEVICE, "| weight decay from 04:", a.weight_decay, "| atypical validation images:", atypical.sum())


def atypicality_weights(X_raw, y, beta):
    # weight 1 + beta * rank, rank = 0 (most typical) to 1 (least typical) within the class
    t = np.concatenate([typicality(np.asarray(X_raw(i)), y[i:i + CHUNK]) for i in range(0, len(y), CHUNK)])
    rank = np.empty(len(y))
    for c in range(N_CLASSES):
        members = np.flatnonzero(y == c)
        order = members[np.argsort(-t[members])]
        rank[order] = np.arange(len(order)) / max(len(order) - 1, 1)
    w = 1 + beta * rank
    return w / w.mean()


def refined_rule(base_val_top1):
    return lambda r: r["val_atypical_top1"] if r["val_top1"] >= base_val_top1 - MAX_VAL_DROP else -np.inf


def gain_ci(correct_ref, correct_base):
    # paired bootstrap of the atypical top-1 difference
    diff = (correct_ref.astype(float) - correct_base)[atypical]
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(DRAWS)]
    return diff.mean(), *np.percentile(boot, [2.5, 97.5])


# 2. Compare the settings on train_small, against A's code on train_small
raw_small = load("train_small")[0]
model, history = train(linear, X_small, y_small, TUNE_EPOCHS, lambda r: r["val_top1"], val, a.weight_decay,
                       val_masks={"atypical": atypical})
base_correct = predict(model, val[0])[:, 0] == val[1]
base = history[history.kept].iloc[0]
grid = [{"setting": "baseline (classifier A on train_small)", "beta": 0.0, "gamma": 0.0,
         "kept_epoch": int(base.epoch), "val_top1": base.val_top1, "val_atypical_top1": base.val_atypical_top1}]
for name, beta, gamma in SETTINGS:
    weights = atypicality_weights(lambda i: raw_small[i:i + CHUNK], y_small, beta) if beta > 0 else None
    model, history = train(linear, X_small, y_small, TUNE_EPOCHS, refined_rule(base.val_top1), val,
                           a.weight_decay, weights, gamma, {"atypical": atypical})
    kept = history[history.kept].iloc[0]
    gain, lo, hi = gain_ci(predict(model, val[0])[:, 0] == val[1], base_correct)
    grid.append({"setting": name, "beta": beta, "gamma": gamma, "kept_epoch": int(kept.epoch),
                 "val_top1": kept.val_top1, "val_atypical_top1": kept.val_atypical_top1, "atypical_gain": gain,
                 "gain_ci_low": lo, "gain_ci_high": hi, "meets_rule": bool(kept.meets_rule)})
grid = pd.DataFrame(grid)
# best atypical top-1 among the settings that meet the rule (all settings if none does)
candidates = grid[grid.meets_rule.eq(True)] if grid.meets_rule.eq(True).any() else grid.iloc[1:]
chosen = candidates.loc[candidates.val_atypical_top1.idxmax()]
grid["chosen"] = grid.index == chosen.name
grid.to_csv(OUT / "refine_grid.csv", index=False)
print(grid.round(4).to_string(index=False))

# 3. The chosen setting on the full training split, against classifier A from 04
X_full, _, train_idx = training_split()
weights = (atypicality_weights(lambda i: X_full[train_idx[i:i + CHUNK]], y_train, chosen.beta)
           if chosen.beta > 0 else None)
t0 = time.time()
model, history = train(linear, X_train, y_train, FULL_EPOCHS, refined_rule(a.val_top1), val, a.weight_decay,
                       weights, chosen.gamma, {"atypical": atypical})
kept = history[history.kept].iloc[0]
final = {"setting": chosen.setting, "beta": chosen.beta, "gamma": chosen.gamma, "kept_epoch": int(kept.epoch),
         "meets_rule": bool(kept.meets_rule), "train_seconds": time.time() - t0, "device": DEVICE}
history.to_csv(OUT / "refine_history.csv", index=False)

# 4. Score once on both test sets
for s in SETS:
    X, y = standardised(s, mean, std)
    top5 = predict(model, X)
    save_predictions("refined", s, y, top5)
    final[f"{s}_top1"], final[f"{s}_top5"] = accuracy(top5, y)
gain, lo, hi = gain_ci(read_predictions("refined", "val")[:, 0] == val[1],
                       read_predictions("linear", "val")[:, 0] == val[1])
final |= {"val_atypical_gain_over_A": gain, "gain_ci_low": lo, "gain_ci_high": hi}
pd.DataFrame([final]).to_csv(OUT / "refine_final.csv", index=False)
print(pd.Series(final).to_string())


# 5. Figure: test set 2 accuracy by typicality decile, classifier A vs refined
y1, y2 = load("test1")[1], load("test2")[1]
t1, t2 = typicality(load("test1")[0], y1), typicality(load("test2")[0], y2)
bins2 = np.digitize(t2, np.quantile(t1, np.linspace(0, 1, 11))[1:-1])
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for model_name, style in (("linear", "o--"), ("refined", "o-")):
    c2 = read_predictions(model_name, "test2")[:, 0] == y2
    ax.plot(range(1, 11), [c2[bins2 == b].mean() for b in range(10)], style,
            color=BLUE if model_name == "linear" else ORANGE, label=MODELS[model_name])
ax.set_xticks(range(1, 11))
ax.set_xlabel("typicality decile of test set 1 (1 = least typical)")
ax.set_ylabel("top-1 on test set 2")
ax.set_title("Test set 2 by typicality: classifier A vs refined")
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_refinement.png", dpi=150, bbox_inches="tight")
plt.close()
