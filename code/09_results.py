# 09_results.py - results table for every classifier and paired bootstrap comparisons against classifier A

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import OUT, SEED, MODELS, BLUE, ORANGE, load, accuracy, read_predictions

plt.rcParams.update({"font.size": 11})

DRAWS = 2000
rng = np.random.default_rng(SEED)
y = {s: load(s)[1] for s in ("val", "test1", "test2")}
dup = pd.read_csv(OUT / "audit_test1_duplicates.csv").duplicate.to_numpy()
models = [m for m in MODELS if (OUT / f"pred_{m}_test2.csv").exists()]
top5 = {m: {s: read_predictions(m, s) for s in y} for m in models}
correct = {m: {s: (top5[m][s][:, 0] == y[s]).astype(float) for s in y} for m in models}


def ci(values):
    return tuple(np.percentile(values, [2.5, 97.5]))


def image_draws(n):
    return [rng.integers(0, n, n) for _ in range(DRAWS)]


def class_draws(labels):
    rows_of = [np.flatnonzero(labels == k) for k in range(labels.max() + 1)]
    return [np.concatenate([rows_of[k] for k in rng.integers(0, len(rows_of), len(rows_of))]) for _ in range(DRAWS)]


# 1. Results table
p = pd.read_csv(OUT / "practice_results.csv")
rows = [{"model": "Logistic regression (practice, EfficientFormerV2-S0)", "test1_top1": p.top1[0],
         "test1_top5": p.top5[0], "test2_top1": p.top1[1], "test2_top5": p.top5[1]}]
draws = {t: image_draws(len(y[t])) for t in ("test1", "test2")}
for m in models:
    row = {"model": MODELS[m]}
    for s in y:
        row[f"{s}_top1"], row[f"{s}_top5"] = accuracy(top5[m][s], y[s])
    row["gap_ci_low"], row["gap_ci_high"] = ci([correct[m]["test1"][i].mean() - correct[m]["test2"][j].mean()
                                                for i, j in zip(draws["test1"], draws["test2"])])
    row["test1_top1_without_duplicates"] = correct[m]["test1"][~dup].mean()
    rows.append(row)
table = pd.DataFrame(rows)
table["gap_top1"] = table.test1_top1 - table.test2_top1
table["gap_without_duplicates"] = table.test1_top1_without_duplicates - table.test2_top1
table.to_csv(OUT / "results_table.csv", index=False)
print(table.round(4).to_string(index=False))


# 2. Paired comparisons with classifier A, resampling images or whole classes
rows = []
for scheme, make in (("image", lambda t: image_draws(len(y[t]))), ("class", lambda t: class_draws(y[t]))):
    draws = {t: make(t) for t in ("test1", "test2")}
    for other in [m for m in models if m != "linear"]:
        boot = {}
        for t in ("test1", "test2"):
            d = correct[other][t] - correct["linear"][t]
            boot[t] = np.array([d[i].mean() for i in draws[t]])
            rows.append({"bootstrap": scheme, "comparison": f"{MODELS[other]} - {MODELS['linear']}",
                         "quantity": f"top-1 difference, {t}", "estimate": d.mean(), **dict(zip(
                             ("ci_low", "ci_high"), ci(boot[t])))})
        est = rows[-2]["estimate"] - rows[-1]["estimate"]
        rows.append({"bootstrap": scheme, "comparison": f"{MODELS[other]} - {MODELS['linear']}",
                     "quantity": "gap difference", "estimate": est,
                     **dict(zip(("ci_low", "ci_high"), ci(boot["test1"] - boot["test2"])))})
paired = pd.DataFrame(rows)
paired["excludes_zero"] = (paired.ci_low > 0) | (paired.ci_high < 0)
for col in ("estimate", "ci_low", "ci_high"):
    paired[col] *= 100
paired.to_csv(OUT / "results_paired.csv", index=False)
print(paired.round(2).to_string(index=False))


# 3. Figure: top-1 on both test sets, with the gap's 95% interval
main = table.iloc[1:].reset_index(drop=True)
x = np.arange(len(main))
fig, ax = plt.subplots(figsize=(8.5, 4))
for off, t, color, label in ((-0.2, "test1", BLUE, "test set 1 (ImageNet val)"),
                             (0.2, "test2", ORANGE, "test set 2 (ImageNetV2)")):
    ax.bar(x + off, main[f"{t}_top1"], 0.4, color=color, label=label)
    for i, v in enumerate(main[f"{t}_top1"]):
        ax.text(i + off, v + 0.01, "%.1f" % (100 * v), ha="center", fontsize=9)
for i, r in main.iterrows():
    ax.text(i, 0.05, "gap %.1f\n(%.1f to %.1f)" % (100 * r.gap_top1, 100 * r.gap_ci_low, 100 * r.gap_ci_high),
            ha="center", fontsize=8, color="white")
ax.set_xticks(x, [m.split(" (")[0] for m in main.model], fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_ylabel("top-1 accuracy")
ax.legend(frameon=False, loc="upper center", ncol=2)
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_results.png", dpi=150, bbox_inches="tight")
plt.close()
