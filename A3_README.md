# CSCI446/946 Assignment 3, Group 4: image analytics with pretrained deep features

## Setup

The scripts run from the root of a clone of https://github.com/huggingface/pytorch-image-models, with
our `validate.py` copied over timm's and `code/` beside it. Python 3.11:

    conda create -n csci946-a3 -c conda-forge --override-channels python=3.11
    conda activate csci946-a3
    pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128   # or .../whl/cpu
    pip install -r A3_requirements.txt
    pip install -e .

Put the supplied files in `data/raw/`: `{train,val}_efficientformerv2_s0.snap_dist_in1k.csv` and
`{train,val,v2}_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv.zip`. The practice task's ImageNetV2 file
is written by our `validate.py` (it saves the 176 features before the head):

    python validate.py <path>/imagenetv2-matched-frequency-format-val --device cpu --model efficientformerv2_s0.snap_dist_in1k

## Run order

Run from the repository root, in file order 01 → 10 (`08b` after `08`). All outputs go to `data/output/`.

| script | what it does | writes |
|---|---|---|
| `01_practice_classifier.py` | practice task: logistic regression on EfficientFormerV2-S0 features (2.5 h) | `practice_results.csv` |
| `02_prepare_data.py` | EVA-02 CSVs to `data/processed/`; our validation set and train_small (25 min, once) | `prepare_summary.csv` |
| `03_integrity_audit.py` | split checks and exact feature duplicates | `audit_*.csv` |
| `04_linear_classifier.py` | classifier A: regularised multinomial logistic regression | `linear_*`, `pred_linear_*` |
| `05_mlp_classifier.py` | classifier B: one-hidden-layer MLP | `mlp_*`, `pred_mlp_*` |
| `06_nearest_centroid.py` | classifier C: nearest class centroid | `centroid_*`, `pred_centroid_*` |
| `07_generalisation_analysis.py` | why test set 2 is harder; `--v2-images <folder>` adds the image figure | `gen_*`, `fig_gen_*` |
| `08_refinement.py` | refines A for atypical images and tests the hypothesis | `refine_*`, `pred_refined_*` |
| `08b_combined_classifier.py` | averages A's and C's probabilities, weight chosen on our validation set | `combined_*` |
| `09_results.py` | results table and paired bootstrap comparisons | `results_*`, `fig_results.png` |
| `10_report_figures.py` | remaining report figures and statistics | `figx_*`, `report_*` |

`SEED = 7` everywhere. `04`, `05`, `08` and `08b` use an NVIDIA GPU when there is one (the device is recorded in
`*_final.csv`) and need about 5 GB of RAM. For `07 --v2-images`, give the folder that holds
`imagenetv2-matched-frequency-format-val`. Models are tuned on our validation set only, and each test set is
scored once per final model.
