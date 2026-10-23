# LCE-Net Pruning Project

## Folder layout

```
LCE-Net-Project/
├── dataset/
│   └── Potato/                    1,500 images, 3 classes (500 each)
│       ├── Potato___Early_blight/
│       ├── Potato___Late_blight/
│       └── Potato___healthy/
├── lcenet_original/
│   └── effnet.py                  UNMODIFIED code from github.com/moufamiao/LCE-Net
│                                   (only data_dir changed to '../dataset/Potato' so it can find the data here)
├── baseline_lcenet.py              LCE-Net baseline (same model/optimizer/LRs as the repo, bug fixes optional)
├── strategy_a.py                   4-phase pruning: LAMP -> surgery -> Taylor -> KD, target < 50,000 params
└── README.md                       this file
```

## Setup

```
pip install torch torchvision torch-pruning scikit-learn matplotlib
```

## Run order

```
cd LCE-Net-Project

# 1. Train the baseline (creates best_model_cbam.pth in this folder)
python baseline_lcenet.py --epochs 50

# 2. Prune it down to under 50,000 parameters
python strategy_a.py --ckpt best_model_cbam.pth --split group
```

`strategy_a.py` needs the checkpoint from step 1 (or from `lcenet_original/effnet.py`, which saves the
same `best_model_cbam.pth` format) as its teacher model. `--split group` is a leak-free split; pass
`--split repo` to use the original repo's exact 80/20 proportions instead.

Both scripts now use a **train/val/test** split (default 70/15/15, not the repo's train/val only).
VAL is used to pick the best checkpoint / monitor progress. TEST is held out completely and only
reported once, at the end - that's the number to trust. `baseline_lcenet.py` prints both at the end;
`strategy_a.py` prints both after every pruning phase, plus a summary table.

## What's what

- **`lcenet_original/effnet.py`** — the code exactly as published in the repo, so you can diff against it
  or reproduce their reported number. Only the dataset path was changed to point at `dataset/Potato`.
- **`baseline_lcenet.py`** — the same LCE-Net architecture, freezing scheme and learning rates, as a
  reusable module that `strategy_a.py` imports from. It optionally fixes two bugs in the original (shared
  train/val dataset object silently disabling augmentation; no leak-free split option).
- **`strategy_a.py`** — the actual pruning pipeline. See its docstring for the four phases.

## Caveat

I have not personally run the full 50-epoch baseline training or the full pruning run on this exact
checkpoint end-to-end — both were smoke-tested on a shrunken setup (fewer images, lower resolution, 1-3
epochs per phase) to confirm the code runs and hits the target parameter count. Actual accuracy numbers
depend on your real training run.


## Why test accuracy isn't 100% (and what changed to improve it)

With a leak-free test set (photos the model never saw in any form), a small gap between val and test
accuracy is normal and expected - it means the split is honest, not that something is broken. The
current recipe (v2):

- Unfreezes `features[6]` **and** `features[7]` (was just `features[6]`), with a discriminative learning
  rate per block (deeper = higher LR) via `EfficientNet_CBAM.param_groups()`.
- Trains with AdamW + cosine-annealed LR + weight decay, instead of one flat LR for the whole run.
- Early-stops on val accuracy (`--patience`, default 12) and always reloads the best checkpoint.
- Evaluates the final test accuracy with test-time augmentation (`--tta`, on by default): it averages
  the prediction with its horizontally-mirrored prediction. This is a label-blind technique used widely
  in image classification - it reduces single-view noise, it does not "cheat".

No training recipe can promise a specific accuracy number on real photographs - if two specific leaves
genuinely look alike, no model will always get them right, and that's a property of the data, not a bug.
`--finetune_layers "5,6,7"` unfreezes even more of the backbone if you want to push further; going much
beyond that on only ~1,000 training images risks overfitting, so watch the val-vs-train gap if you do.


## strategy_b.py — the Taylor-LAMP Hybrid (single-shot alternative)

`strategy_a.py` prunes in 3 distinct sequential steps (LAMP coarse cut -> manual architectural surgery ->
iterative Taylor pruning). `strategy_b.py` is a different method: it scores every filter in the whole
network ONCE, with a formula that is both data-aware (Taylor: real leaf images + gradients) and
layer-safe (LAMP: normalized so every layer always keeps its best filter), then makes a single global cut
straight down to the parameter budget. No manual "delete this layer" surgery step - the oversized
320->1280 head gets pruned away automatically because the same global ranking covers it too.

```
python strategy_b.py --ckpt best_model_cbam.pth --ckpt_finetune_layers 6,7
```

Both scripts hit the same < 50,000 parameter budget from different philosophies - `strategy_a.py` trades
more manual control for more code, `strategy_b.py` trades a single unified formula for needing real data
from the very first pruning decision (Taylor needs gradients from actual images, unlike LAMP alone).
Run both and compare their final TEST accuracy printed at the end of each.
