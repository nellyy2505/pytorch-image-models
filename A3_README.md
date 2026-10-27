# CSCI446/946 Assignment 3: image analytics with pretrained deep features

A clone of https://github.com/huggingface/pytorch-image-models. Our work is `validate.py`, `code/` and `data/`.

## Setup

    conda create -n csci946-a3 -c conda-forge --override-channels python=3.11
    conda activate csci946-a3
    pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128   # or .../whl/cpu
    pip install -r A3_requirements.txt

`data/raw/` (not in git) holds the supplied files:
`{train,val}_efficientformerv2_s0.snap_dist_in1k.csv`, `{train,val,v2}_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv.zip`,
and `v2_efficientformerv2_s0.snap_dist_in1k.csv`, written by our `validate.py` (176-d features before the head):

    python validate.py <path>/imagenetv2-matched-frequency-format-val --device cpu --model efficientformerv2_s0.snap_dist_in1k

## Run order

From the repository root, `01` to `09` in order. `SEED = 7` everywhere. `04`, `05` and `08` use an NVIDIA GPU
when there is one (reported numbers: Quadro T2000, 4 GB) and the CPU otherwise; the device is recorded in
`*_final.csv`, and another device can move results in the second decimal. All outputs go to `data/output/`.

| script | does | time (laptop, Quadro T2000) |
|---|---|---|
| `01_practice_classifier.py` | practice task: logistic regression on EfficientFormerV2-S0 features | 2.5 h (CPU) |
| `02_prepare_data.py` | EVA-02 CSVs to `data/processed/*.npy`; our validation set (50 per class) and train_small (100 per class) | 25 min once |
| `03_integrity_audit.py` | split checks and exact feature duplicates | 3 min |
| `04_linear_classifier.py` | classifier A: multinomial logistic regression with weight decay | 2 min |
| `05_mlp_classifier.py` | classifier B: one-hidden-layer MLP | 15 min |
| `06_nearest_centroid.py` | classifier C: nearest class centroid (CPU) | 15 min |
| `07_generalisation_analysis.py` | why test set 2 is harder (`--v2-images <folder>` adds the image figure) | 2 min |
| `08_refinement.py` | refines classifier A for atypical images and tests the hypothesis | 4 min |
| `09_results.py` | results table and paired comparisons | 1 min |

`code/common.py` holds the shared paths, typicality measure, training loop (A, B, refinement) and prediction format.
Models are tuned and selected on our validation set only; each test set is scored once per final model.
