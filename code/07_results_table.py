# 06_results_table.py - integration: one results table for every classifier, built from the files each script writes
# Run from the repository root after the classifier scripts:  python code/06_results_table.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"


def read(name):
    return pd.read_csv(OUT / name)


rows = []

# Practice task: logistic regression on EfficientFormerV2-S0 features (no validation split was used)
p = read("practice_results.csv")
rows.append({"method": "Logistic regression (practice)", "owner": "Nelly", "features": "EfficientFormerV2-S0, 176-d",
             "val_top1": np.nan, "test1_top1": p.top1[0], "test1_top5": p.top5[0],
             "test2_top1": p.top1[1], "test2_top5": p.top5[1], "script": "01_practice_classifier.py"})

# Classifier A: regularised linear; val score of the epoch kept as the final model
a = read("classifier_a_test_results.csv").set_index("test_set")
a_val = read("classifier_a_full_alpha1e-05_epochs10.csv")
a_val = a_val.loc[a_val.epoch == int(a.loc["test1", "best_epoch"]), "val_top1"].iloc[0]
rows.append({"method": "Regularised linear (A)", "owner": "Andrew", "features": "EVA-02, 1024-d",
             "val_top1": a_val, "test1_top1": a.loc["test1", "top1"], "test1_top5": a.loc["test1", "top5"],
             "test2_top1": a.loc["test2", "top1"], "test2_top5": a.loc["test2", "top5"],
             "script": "classifier_a_linear.py, classifier_a_evaluate.py"})

# Classifier B: small MLP; val score is the best epoch of the final run
b = read("mlp_results.csv")
b_val = read("mlp_final_history.csv").val_top1.max()
rows.append({"method": "Small MLP (B)", "owner": "Ban", "features": "EVA-02, 1024-d",
             "val_top1": b_val, "test1_top1": b.top1[0], "test1_top5": b.top5[0],
             "test2_top1": b.top1[1], "test2_top5": b.top5[1], "script": "03_mlp_classifier.py"})

# Classifier C: nearest centroid (the chosen variant is listed in ncm_results.csv)
c = read("ncm_results.csv")
for model, part in c.groupby("model", sort=False):
    part = part.set_index("set")
    rows.append({"method": "Nearest centroid (C), %s" % model, "owner": "Trung", "features": "EVA-02, 1024-d",
                 "val_top1": part.loc["val", "top1"], "test1_top1": part.loc["test1", "top1"],
                 "test1_top5": part.loc["test1", "top5"], "test2_top1": part.loc["test2", "top1"],
                 "test2_top5": part.loc["test2", "top5"], "script": "03_nearest_centroid.py"})

# Refined classifier (part c): add a row here once its results file exists

table = pd.DataFrame(rows)
table["gap_top1"] = table.test1_top1 - table.test2_top1

# test1 without the images whose features exactly match a training image (from the generalisation analysis)
g = read("gen_overall_accuracy.csv")
g = g[g.subset == "test1 without training duplicates"].set_index("classifier").test1_top1
names = {"Regularised linear (A)": "Regularised linear", "Small MLP (B)": "Small MLP"}
table["test1_top1_no_dups"] = [g.get(names.get(m, "Nearest centroid" if m.startswith("Nearest") else ""), np.nan)
                               for m in table.method]

table.round(4).to_csv(OUT / "results_integrated.csv", index=False)
print(table.drop(columns="script").round(4).to_string(index=False))

# Figure: top-1 on both test sets for every main-task classifier
main = table[table.features.str.startswith("EVA")]
x = np.arange(len(main))
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.bar(x - 0.2, main.test1_top1, 0.4, label="test set 1 (ImageNet val)", color="#2a78d6")
ax.bar(x + 0.2, main.test2_top1, 0.4, label="test set 2 (ImageNetV2)", color="#eb6834")
for i, (t1, t2) in enumerate(zip(main.test1_top1, main.test2_top1)):
    ax.text(i - 0.2, t1 + 0.005, "%.1f" % (100 * t1), ha="center", fontsize=9)
    ax.text(i + 0.2, t2 + 0.005, "%.1f" % (100 * t2), ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([m.split(",")[0] for m in main.method], fontsize=9)
ax.set_ylim(0.7, 0.95)
ax.set_ylabel("top-1 accuracy")
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_results_integrated.png", dpi=150, bbox_inches="tight")
