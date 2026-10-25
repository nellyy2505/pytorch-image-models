# 08_paired_comparisons.py - paired bootstrap intervals for accuracy differences between classifiers.
# Each draw resamples the same images for both models; gap differences resample the two test sets independently,
# with a class-level bootstrap as a sensitivity check. Trains nothing; run after the classifier scripts and 06_refinement.py.
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"
SEED, DRAWS = 7, 2000
rng = np.random.default_rng(SEED)


# 1. Top-1 correctness per image, in the same row order for every model
def classifier_a(t):
    p = pd.read_csv(OUT / f"classifier_a_{t}_predictions.csv", usecols=["true_label", "predicted_label"])
    return p.true_label.to_numpy(), (p.true_label == p.predicted_label).to_numpy()


def classifier_b(t):
    p = pd.read_csv(OUT / "mlp_predictions.csv")
    p = p[p.test_set == t].sort_values("row")
    return p.label.to_numpy(), (p.label == p.pred).to_numpy()


def classifier_c(t):
    p = pd.read_csv(OUT / f"ncm_predictions_{t}.csv").sort_values("row")
    return p.label.to_numpy(), (p.label == p.pred1).to_numpy()


def refinement(run):
    def read(t):
        p = pd.read_csv(OUT / "refine_predictions.csv")
        p = p[(p.run == run) & (p.test_set == t)].sort_values("row")
        return p.label.to_numpy(), (p.label == p.top1).to_numpy()
    return read


runs = pd.read_csv(OUT / "refine_results.csv").run.tolist()     # [baseline, chosen refinement]
models = {"A": classifier_a, "B": classifier_b, "C": classifier_c,
          "refine_baseline": refinement(runs[0]), "refine_R1": refinement(runs[1])}
labels, correct = {}, {}
for t in ("test1", "test2"):
    correct[t] = {}
    for name, reader in models.items():
        y, c = reader(t)
        if t in labels and not np.array_equal(labels[t], y):
            raise ValueError(f"{name} {t}: rows are not in the same order as the other models")
        labels[t], correct[t][name] = y, c.astype(float)

pairs = [("A", "B"), ("A", "C"), ("refine_baseline", "refine_R1")]


# 2. Bootstrap draws: image-level and class-level
def image_draws(n):
    return [rng.integers(0, n, n) for _ in range(DRAWS)]


def class_draws(y):
    rows_of = [np.flatnonzero(y == k) for k in range(y.max() + 1)]
    picks = [rng.integers(0, len(rows_of), len(rows_of)) for _ in range(DRAWS)]
    return [np.concatenate([rows_of[k] for k in pick]) for pick in picks]


def ci(values):
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


rows = []
for scheme, make in (("image", lambda t: image_draws(len(labels[t]))), ("class", lambda t: class_draws(labels[t]))):
    draws = {t: make(t) for t in ("test1", "test2")}
    for x, y in pairs:
        boot = {}
        for t in ("test1", "test2"):
            d = correct[t][x] - correct[t][y]
            boot[t] = np.array([d[idx].mean() for idx in draws[t]])
            lo, hi = ci(boot[t])
            rows.append({"bootstrap": scheme, "comparison": f"{x} - {y}", "quantity": f"top-1 difference, {t}",
                         "estimate": d.mean(), "ci_low": lo, "ci_high": hi, "excludes_zero": not lo <= 0 <= hi})
        # difference in gap = (x1 - x2) - (y1 - y2) = (x1 - y1) - (x2 - y2); test sets resampled independently
        est = (correct["test1"][x] - correct["test1"][y]).mean() - (correct["test2"][x] - correct["test2"][y]).mean()
        lo, hi = ci(boot["test1"] - boot["test2"])
        rows.append({"bootstrap": scheme, "comparison": f"{x} - {y}", "quantity": "gap difference",
                     "estimate": est, "ci_low": lo, "ci_high": hi, "excludes_zero": not lo <= 0 <= hi})

# 3. Gap of each model with an image-level bootstrap, for reference
for name in models:
    g = [correct["test1"][name][i].mean() - correct["test2"][name][j].mean()
         for i, j in zip(image_draws(len(labels["test1"])), image_draws(len(labels["test2"])))]
    est = correct["test1"][name].mean() - correct["test2"][name].mean()
    lo, hi = ci(g)
    rows.append({"bootstrap": "image", "comparison": name, "quantity": "gap", "estimate": est,
                 "ci_low": lo, "ci_high": hi, "excludes_zero": not lo <= 0 <= hi})

result = pd.DataFrame(rows)
for col in ("estimate", "ci_low", "ci_high"):
    result[col] = 100 * result[col]                                  # percentage points
result.to_csv(OUT / "paired_comparisons.csv", index=False)
print(result.round(2).to_string(index=False))
