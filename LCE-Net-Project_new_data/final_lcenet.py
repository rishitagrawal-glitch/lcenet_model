"""
final_lcenet.py  -  Improved LCE-Net Training Script
=====================================================
Improvements over baseline_lcenet.py:
  1. Stronger data augmentation  (RandomErasing, GaussianBlur, stronger ColorJitter)
  2. Focal Loss                  (handles class imbalance better than CrossEntropyLoss)
  3. AdamW + Weight Decay        (reduces overfitting)
  4. CosineAnnealing LR Scheduler(smooth learning rate decay)
  5. Label Smoothing             (improves generalization)
  6. Train CBAM                  (attention is now actually learned, not frozen at random init)
  7. Unfreeze more layers        (features[5], [6], [7] + classifier + cbam)
  8. Confusion matrix saved      (visual result)
  9. Per-class accuracy printed  (easy to see weak classes)

Usage:
    python final_lcenet.py --data ./dataset_extracted --epochs 60 --oversample
"""

import argparse, os, re
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision import datasets, models, transforms

# ============================================================
#  1. LITE-CBAM ATTENTION MODULE (same as baseline)
# ============================================================
class Lite_CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        mid = max(1, channels // reduction)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False), nn.ReLU(),
            nn.Conv2d(mid, channels, 1, bias=False), nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)
        sa = self.sa(torch.cat([
            torch.max(x, 1, keepdim=True)[0],
            torch.mean(x, 1, keepdim=True)
        ], 1))
        return x * sa


# ============================================================
#  2. EFFICIENTNET-B0 + CBAM MODEL
# ============================================================
class EfficientNet_CBAM(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, unfreeze_stages=(5, 6, 7)):
        super().__init__()
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_b0(weights=w)
        self.features   = base.features
        self.cbam       = Lite_CBAM(channels=320)
        self.head       = self.features[8]       # 320 -> 1280 conv
        self.pool       = base.avgpool
        self.dropout    = nn.Dropout(p=0.3)      # slightly higher dropout
        self.classifier = nn.Linear(1280, num_classes)

        # Freeze everything first
        for p in self.parameters():
            p.requires_grad = False

        # Unfreeze selected stages + cbam + classifier
        for name, p in self.named_parameters():
            stage_ok = any(f"features.{s}" in name for s in unfreeze_stages)
            if stage_ok or "classifier" in name or name.startswith("cbam"):
                p.requires_grad = True

    def forward(self, x):
        for i in range(8):
            x = self.features[i](x)
        x = self.cbam(x)
        x = self.head(x)
        x = torch.flatten(self.pool(x), 1)
        return self.classifier(self.dropout(x))


# ============================================================
#  3. FOCAL LOSS (better than CrossEntropy for imbalanced data)
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples and focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=2 is the standard. alpha handles class imbalance.
    """
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.05):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ============================================================
#  4. DATA SPLIT (same leak-free group split as baseline)
# ============================================================
def group_of(path):
    """Healthy images: 'Potato_healthy-<id>-_0_<n>.jpg' -> group by source photo id."""
    f = os.path.basename(path)
    m = re.match(r'(Potato_healthy-\d+)-_0_\d+', f)
    return m.group(1) if m else f


def three_way_split(ds, split='group', seed=42, fracs=(0.70, 0.15, 0.15)):
    n = len(ds)
    if split == 'repo':
        g  = torch.Generator().manual_seed(seed)
        tr, rest = random_split(range(n), [int(0.8*n), n - int(0.8*n)], generator=g)
        g2 = torch.Generator().manual_seed(seed + 1)
        va, te = random_split(rest.indices,
                              [len(rest)//2, len(rest) - len(rest)//2], generator=g2)
        return list(tr.indices), \
               [rest.indices[i] for i in va.indices], \
               [rest.indices[i] for i in te.indices]

    groups = [group_of(p) for p, _ in ds.samples]
    ug     = sorted(set(groups))
    np.random.RandomState(seed).shuffle(ug)
    n1, n2 = int(fracs[0]*len(ug)), int((fracs[0]+fracs[1])*len(ug))
    tr_g, va_g = set(ug[:n1]), set(ug[n1:n2])
    tr, va, te = [], [], []
    for i, g in enumerate(groups):
        (tr if g in tr_g else va if g in va_g else te).append(i)
    return tr, va, te


# ============================================================
#  5. DATA LOADERS WITH STRONG AUGMENTATION
# ============================================================
def make_loaders(root, split='group', res=224, bs=32, seed=42,
                 oversample=False, limit=None, workers=2):
    norm = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])  # ImageNet stats (better than 0.5/0.5)

    # --- IMPROVED augmentation for training ---
    tf_train = transforms.Compose([
        transforms.Resize((res + 32, res + 32)),
        transforms.RandomCrop(res),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # random patch erasing
    ])

    # --- Evaluation: just resize + normalize ---
    tf_eval = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        norm,
    ])

    ds_tr = datasets.ImageFolder(root, tf_train)
    ds_ev = datasets.ImageFolder(root, tf_eval)

    tr, va, te = three_way_split(ds_tr, split, seed)
    if limit:
        tr = tr[:limit]
        va = va[:max(1, limit//4)]
        te = te[:max(1, limit//4)]

    print(f'split={split}: train={len(tr)}  val={len(va)}  test={len(te)}')
    print(f'classes: {ds_tr.classes}')

    labels = [ds_tr.samples[i][1] for i in tr]

    if oversample:
        counts  = np.bincount(labels)
        w       = 1.0 / counts
        sampler = WeightedRandomSampler([w[l] for l in labels], len(labels), True)
        train_loader = DataLoader(Subset(ds_tr, tr), bs, sampler=sampler,
                                  num_workers=workers, pin_memory=True)
    else:
        train_loader = DataLoader(Subset(ds_tr, tr), bs, shuffle=True,
                                  num_workers=workers, pin_memory=True,
                                  drop_last=len(tr) > bs)

    val_loader  = DataLoader(Subset(ds_ev, va), bs, shuffle=False,
                             num_workers=workers, pin_memory=True)
    test_loader = DataLoader(Subset(ds_ev, te), bs, shuffle=False,
                             num_workers=workers, pin_memory=True)

    return train_loader, val_loader, test_loader, ds_tr.classes


# ============================================================
#  6. EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(model, loader, dev, return_preds=False):
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        logits = model(x.to(dev))
        yp += logits.argmax(1).cpu().tolist()
        yt += y.tolist()
    acc = float(np.mean(np.array(yt) == np.array(yp))) if yt else float('nan')
    return (acc, yt, yp) if return_preds else acc


# ============================================================
#  7. MAIN TRAINING LOOP
# ============================================================
def main():
    ap = argparse.ArgumentParser(description='Improved LCE-Net Trainer')
    ap.add_argument('--data',         default='./dataset_extracted')
    ap.add_argument('--split',        choices=['group', 'repo'], default='group')
    ap.add_argument('--epochs',       type=int,   default=60)
    ap.add_argument('--bs',           type=int,   default=32)
    ap.add_argument('--res',          type=int,   default=224)
    ap.add_argument('--lr',           type=float, default=5e-4)
    ap.add_argument('--oversample',   action='store_true',
                    help='Balance classes via WeightedRandomSampler')
    ap.add_argument('--focal_gamma',  type=float, default=2.0,
                    help='Focal loss gamma (0 = standard cross-entropy)')
    ap.add_argument('--no_pretrained',action='store_true')
    ap.add_argument('--out',          default='best_model_final.pth')
    ap.add_argument('--limit',        type=int, default=None,
                    help='Debug: use only N images')
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {dev}')

    # --- Data ---
    tr, va, te, classes = make_loaders(
        a.data, a.split, a.res, a.bs,
        oversample=a.oversample, limit=a.limit
    )
    num_classes = len(classes)

    # --- Model: unfreeze stages 5, 6, 7 + cbam + classifier ---
    model = EfficientNet_CBAM(
        num_classes  = num_classes,
        pretrained   = not a.no_pretrained,
        unfreeze_stages=(5, 6, 7)
    ).to(dev)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total:,}   Trainable: {trainable:,}')

    # --- Compute class weights from training labels (for Focal Loss) ---
    train_labels = [tr.dataset.dataset.samples[i][1] for i in tr.dataset.indices]
    counts       = np.bincount(train_labels, minlength=num_classes).astype(float)
    class_weights= torch.tensor(counts.sum() / (num_classes * counts),
                                dtype=torch.float32).to(dev)
    print(f'Class weights: {class_weights.cpu().numpy().round(3)}')

    # --- Loss: Focal Loss with class weights ---
    criterion = FocalLoss(
        gamma            = a.focal_gamma,
        weight           = class_weights,
        label_smoothing  = 0.05
    )

    # --- Optimizer: AdamW with weight decay ---
    param_groups = [
        {'params': model.features[5].parameters(), 'lr': a.lr * 0.1},
        {'params': model.features[6].parameters(), 'lr': a.lr * 0.2},
        {'params': model.features[7].parameters(), 'lr': a.lr * 0.5},
        {'params': model.cbam.parameters(),         'lr': a.lr},
        {'params': model.classifier.parameters(),   'lr': a.lr},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # --- LR Scheduler: Cosine Annealing ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.epochs, eta_min=1e-6
    )

    # --- Training loop ---
    best_val = -1.0
    print(f'\nTraining for {a.epochs} epochs...\n')

    for ep in range(a.epochs):
        model.train()
        tot_loss, corr, n = 0.0, 0, 0

        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            tot_loss += loss.item() * len(y)
            corr     += (out.argmax(1) == y).sum().item()
            n        += len(y)

        scheduler.step()
        val_acc = evaluate(model, va, dev)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), a.out)

        print(f'Epoch {ep+1:3d}/{a.epochs} | '
              f'Loss {tot_loss/n:.4f} | '
              f'Train Acc {corr/n:.4f} | '
              f'Val Acc {val_acc:.4f} | '
              f'LR {scheduler.get_last_lr()[0]:.2e}',
              flush=True)

    # --- Final evaluation on HELD-OUT TEST SET ---
    print(f'\n{"="*60}')
    print(f'Best VAL accuracy: {best_val:.4f}')
    print(f'Loading best checkpoint: {a.out}')
    model.load_state_dict(torch.load(a.out, map_location=dev))

    from sklearn.metrics import classification_report, confusion_matrix

    test_acc, yt, yp = evaluate(model, te, dev, return_preds=True)
    print(f'\nBest-epoch VAL accuracy  (checkpoint selection): {best_val:.4f}')
    print(f'Held-out  TEST accuracy  (never seen in training): {test_acc:.4f}')

    if yt:
        print('\nTest Set Classification Report:')
        print(classification_report(
            yt, yp,
            labels      = list(range(num_classes)),
            target_names= classes,
            digits      = 4,
            zero_division= 0
        ))

        # --- Per-class accuracy ---
        cm = confusion_matrix(yt, yp)
        print('Per-class accuracy:')
        for i, cls in enumerate(classes):
            cls_total   = cm[i].sum()
            cls_correct = cm[i, i]
            print(f'  {cls:<15} {cls_correct}/{cls_total} = {cls_correct/cls_total:.4f}')

        # --- Save confusion matrix as image ---
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix  (Test Acc = {test_acc:.4f})')
            plt.tight_layout()
            plt.savefig('confusion_matrix_final.png', dpi=150)
            print('\nConfusion matrix saved to: confusion_matrix_final.png')
        except ImportError:
            print('(Install matplotlib + seaborn to save confusion matrix image)')

    print(f'\nDone! Best model saved to: {a.out}')


if __name__ == '__main__':
    main()