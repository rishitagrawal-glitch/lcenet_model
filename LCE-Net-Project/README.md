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

### Using `final_lcenet.py` + `strategy_a.py` (Recommended):
```powershell
cd LCE-Net-Project

# 1. Train the improved model (creates best_model_final.pth)
python final_lcenet.py --data ./dataset_extracted --epochs 60 --oversample

# 2. Prune it down to under 50,000 parameters with 3-phase pruning
python strategy_a.py --data ./dataset_extracted --ckpt best_model_final.pth
```

### Or using baseline LCE-Net:
```powershell
# 1. Train the baseline (creates best_model_cbam.pth in this folder)
python baseline_lcenet.py --epochs 50

# 2. Prune baseline down to under 50,000 parameters
python strategy_a.py --data ./dataset/Potato --ckpt best_model_cbam.pth --split group
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
