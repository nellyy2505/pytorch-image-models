# 08b_combined_classifier.py - classifier A + C: a weighted average of A's and C's class probabilities

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from common import OUT, N_CLASSES, CHUNK, SETS, DEVICE, BLUE, ORANGE, load, accuracy, centroid_scores

plt.rcParams.update({"font.size": 11})

WEIGHTS = np.round(np.linspace(0, 1, 11), 1)       # weight on A: 0 is C alone, 1 is A alone


# 1. The final A and C, saved by 04 and 06
saved = torch.load(OUT / "linear_model.pt")
linear = torch.nn.Linear(1024, N_CLASSES)
linear.load_state_dict(saved["state"])
linear.to(DEVICE).eval()
centroid = np.load(OUT / "centroid_model.npz")
if str(centroid["model"]) != "raw, Euclidean":
    raise ValueError(f"08b scores C on raw features, but 06 chose {centroid['model']}")


def scores(name):
    # A's probabilities and C's raw scores for every image of one set
    X, y = load(name)
    with torch.no_grad():
        p_a = np.vstack([linear(((torch.from_numpy(X[i:i + CHUNK]) - saved["mean"]) / saved["std"]).to(DEVICE))
                         .softmax(dim=1).cpu().numpy() for i in range(0, len(X), CHUNK)])
    s_c = np.vstack([centroid_scores(X[i:i + CHUNK], centroid["centroids"], bool(centroid["cosine"]))
                     .astype(np.float32) for i in range(0, len(X), CHUNK)])
    return p_a, s_c, y


def top5(p):
    return np.argsort(-p, axis=1)[:, :5]


# 2. Turn C's scores into probabilities: one temperature, fitted by log loss on our validation set
p_a, s_c, y_val = scores("val")
log_loss = lambda log_t: -log_softmax(s_c / 10 ** log_t, axis=1)[np.arange(len(y_val)), y_val].mean()
fit = minimize_scalar(log_loss, bounds=(-2, 6), method="bounded")
temperature = 10 ** fit.x
p_c = softmax(s_c / temperature, axis=1)
print("C temperature %.4g, validation log loss %.4f" % (temperature, fit.fun))

# 3. Choose the weight on A by validation top-1
tuning = pd.DataFrame([dict(zip(("val_top1", "val_top5"), accuracy(top5(w * p_a + (1 - w) * p_c), y_val)),
                            weight_a=w) for w in WEIGHTS])
tuning.to_csv(OUT / "combined_tuning.csv", index=False)
w = tuning.loc[tuning.val_top1.idxmax(), "weight_a"]
print(tuning.round(5).to_string(index=False))
final = {"weight_a": w, "temperature_c": temperature}

# 4. Score once on both test sets
for s in SETS:
    if s != "val":
        p_a, s_c, y = scores(s)
        p_c = softmax(s_c / temperature, axis=1)
    else:
        y = y_val
    pred = top5(w * p_a + (1 - w) * p_c)
    final[f"{s}_top1"], final[f"{s}_top5"] = accuracy(pred, y)
pd.DataFrame([final]).to_csv(OUT / "combined_final.csv", index=False)
print(pd.Series(final).to_string())

fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.plot(tuning.weight_a, 100 * tuning.val_top1, "o-", color=BLUE)
ax.plot(w, 100 * tuning.val_top1.max(), "o", ms=11, mfc="none", mec=ORANGE, mew=2, label="chosen weight")
ax.set_xlabel("weight on A (0 = C alone, 1 = A alone)")
ax.set_ylabel("validation top-1 (%)")
ax.set_title("A + C: weight on A's probabilities")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_combined_weight.png", dpi=150, bbox_inches="tight")
plt.close()
