"""
baseline_lcenet.py  -  Baseline model: LCE-Net = EfficientNet-B0 + Lite-CBAM (moufamiao/LCE-Net),
                       with a stronger (but still honest) fine-tuning recipe than the original repo.

Three-way split (train/val/test), same as before - see three_way_split() below.
  * TRAIN  used to update weights.
  * VAL    used to pick the best checkpoint and drive early stopping. Seen many times during training.
  * TEST   held out completely until the very end. Printed ONCE. That's the number to report.

WHAT CHANGED vs the previous version, and why (this is what actually moves test accuracy - there is no
trick that guarantees a specific number, only a training recipe that generalizes better):

  1. --finetune_layers "6,7"  (was hard-coded to just "6")
     The repo only unfreezes features[6] (one MBConv block) + the classifier - everything else, including
     features[7] which sits right before your CBAM, stays frozen at generic ImageNet features. Unfreezing
     one more block lets the network adapt features closer to the classifier to YOUR blight textures,
     which is exactly where Early vs Late blight confusion happens.
  2. Discriminative learning rates per unfrozen block (deeper = higher LR) instead of one shared LR,
     so early unfrozen blocks (still fairly general) move slowly and don't get destroyed.
  3. Cosine-annealed LR + weight decay (AdamW) instead of a flat LR for the whole run - smoother
     convergence, less late-training overfitting to train-set noise.
  4. --patience early stopping on val accuracy (default 12 epochs) - stops before the model starts
     memorizing train-set idiosyncrasies past the point where it generalizes.
  5. --tta test-time augmentation (horizontal-flip averaging) at FINAL evaluation only - this is a
     standard, honest technique (average the prediction with its mirror image's prediction), not
     something that touches training or peeks at labels. Usually adds a fraction of a percent by
     smoothing out single-view noise. It will not turn a genuinely ambiguous leaf into a correct answer.

None of this can promise a specific accuracy number on your particular photos. If Early and Late blight
truly look identical in some of your images, no amount of training fixes that - that's not a bug, it's
the actual difficulty of the problem.

Usage:  python baseline_lcenet.py --data ./dataset/Potato --epochs 60
"""
import argparse, os, re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision import datasets, models, transforms


class Lite_CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        mid = max(1, channels // reduction)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, mid, 1, bias=False), nn.ReLU(),
                                nn.Conv2d(mid, channels, 1, bias=False), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ca(x)
        sa = self.sa(torch.cat([torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)], 1))
        return x * sa


class EfficientNet_CBAM(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, train_cbam=False, finetune_layers=(6,)):
        super().__init__()
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_b0(weights=w)
        self.features = base.features
        self.cbam = Lite_CBAM(channels=320)
        self.head = self.features[8]                     # Conv2d(320 -> 1280); alias kept for checkpoint parity
        self.pool = base.avgpool
        self.dropout = base.classifier[0]
        self.classifier = nn.Linear(1280, num_classes)
        self.finetune_layers = tuple(finetune_layers)
        unlock = tuple(f"features.{i}" for i in self.finetune_layers)
        for p in self.parameters():
            p.requires_grad = False
        for name, p in self.named_parameters():
            if name.startswith(unlock) or "classifier" in name or (train_cbam and name.startswith("cbam")):
                p.requires_grad = True

    def forward(self, x):
        for i in range(8):
            x = self.features[i](x)
        x = self.cbam(x)
        x = self.head(x)
        x = torch.flatten(self.pool(x), 1)
        return self.classifier(self.dropout(x))

    def param_groups(self, base_lr=5e-4):
        """Discriminative LRs: deeper unfrozen blocks get a higher LR than shallower ones; classifier
        highest of all since it's trained from scratch."""
        groups = []
        for rank, i in enumerate(sorted(self.finetune_layers)):
            lr = base_lr * (0.02 ** (1 - (rank + 1) / len(self.finetune_layers)))  # shallow ~1e-5 .. deep ~base_lr
            groups.append({'params': self.features[i].parameters(), 'lr': lr})
        if any(p.requires_grad for p in self.cbam.parameters()):
            groups.append({'params': self.cbam.parameters(), 'lr': base_lr})
        groups.append({'params': self.classifier.parameters(), 'lr': base_lr})
        return groups


# ----------------------------------------------------------------------------- data
def group_of(path):
    """Healthy images are augmented copies 'Potato_healthy-<id>-_0_<n>.jpg' of one source photo."""
    f = os.path.basename(path); m = re.match(r'(Potato_healthy-\d+)-_0_\d+', f)
    return m.group(1) if m else f


def three_way_split(ds, split, seed=42, fracs=(0.70, 0.15, 0.15)):
    """Returns (train_idx, val_idx, test_idx). 'group' keeps augmented siblings of one source photo
    together on one side, so no photo's augmented copies leak across train/val/test. 'repo' mimics the
    original 80/20 proportions (val+test share what used to be the repo's single val split, 50/50)."""
    n = len(ds)
    if split == 'repo':
        g = torch.Generator().manual_seed(seed)
        tr, rest = random_split(range(n), [int(0.8 * n), n - int(0.8 * n)], generator=g)
        g2 = torch.Generator().manual_seed(seed + 1)
        va, te = random_split(rest.indices, [len(rest) // 2, len(rest) - len(rest) // 2], generator=g2)
        return list(tr.indices), [rest.indices[i] for i in va.indices], [rest.indices[i] for i in te.indices]
    groups = [group_of(p) for p, _ in ds.samples]
    ug = sorted(set(groups)); np.random.RandomState(seed).shuffle(ug)
    n1, n2 = int(fracs[0] * len(ug)), int((fracs[0] + fracs[1]) * len(ug))
    tr_g, va_g = set(ug[:n1]), set(ug[n1:n2])
    tr, va, te = [], [], []
    for i, g in enumerate(groups):
        (tr if g in tr_g else va if g in va_g else te).append(i)
    return tr, va, te


def make_loaders(root, split='group', res=224, bs=32, seed=42, oversample=False, limit=None, workers=2):
    norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
    tf_train = transforms.Compose([transforms.Resize((res, res)), transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(p=0.3), transforms.RandomRotation(20),
                                   transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
                                   transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                                   transforms.ToTensor(), norm])
    tf_eval = transforms.Compose([transforms.Resize((res, res)), transforms.ToTensor(), norm])
    ds_tr, ds_ev = datasets.ImageFolder(root, tf_train), datasets.ImageFolder(root, tf_eval)
    tr, va, te = three_way_split(ds_tr, split, seed)
    if limit:
        tr, va, te = tr[:limit], va[:max(1, limit // 4)], te[:max(1, limit // 4)]
    print(f'split={split}: train={len(tr)} val={len(va)} test={len(te)} classes={ds_tr.classes}')
    labels = [ds_tr.samples[i][1] for i in tr]
    if oversample:                                       # the repo's WeightedRandomSampler
        w = 1.0 / np.bincount(labels); sampler = WeightedRandomSampler([w[l] for l in labels], len(labels), True)
        train_loader = DataLoader(Subset(ds_tr, tr), bs, sampler=sampler, num_workers=workers)
    else:
        train_loader = DataLoader(Subset(ds_tr, tr), bs, shuffle=True, num_workers=workers, drop_last=len(tr) > bs)
    val_loader = DataLoader(Subset(ds_ev, va), bs, shuffle=False, num_workers=workers)
    test_loader = DataLoader(Subset(ds_ev, te), bs, shuffle=False, num_workers=workers)
    return train_loader, val_loader, test_loader, ds_tr.classes


@torch.no_grad()
def evaluate(model, loader, dev, return_preds=False, tta=False):
    """tta=True averages softmax over the image and its horizontal mirror - a standard, label-blind
    technique. It cannot fix genuinely ambiguous images; it only reduces single-view noise."""
    model.eval(); yt, yp = [], []
    for x, y in loader:
        x = x.to(dev)
        probs = F.softmax(model(x), 1)
        if tta:
            probs = (probs + F.softmax(model(torch.flip(x, dims=[3])), 1)) / 2
        yp += probs.argmax(1).cpu().tolist(); yt += y.tolist()
    acc = float(np.mean(np.array(yt) == np.array(yp))) if yt else float('nan')
    return (acc, yt, yp) if return_preds else acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./dataset/Potato'); ap.add_argument('--split', choices=['group', 'repo'], default='group')
    ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--res', type=int, default=224); ap.add_argument('--oversample', action='store_true')
    ap.add_argument('--train_cbam', action='store_true'); ap.add_argument('--no_pretrained', action='store_true')
    ap.add_argument('--finetune_layers', default='6,7', help='comma list of features[i] blocks to unfreeze, e.g. "5,6,7"')
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=12, help='stop if val acc does not improve for this many epochs')
    ap.add_argument('--tta', action='store_true', default=True); ap.add_argument('--no_tta', dest='tta', action='store_false')
    ap.add_argument('--out', default='best_model_cbam.pth'); ap.add_argument('--limit', type=int, default=None, help='debug: N images')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr, va, te, classes = make_loaders(a.data, a.split, a.res, a.bs, oversample=a.oversample, limit=a.limit)
    layers = tuple(int(x) for x in a.finetune_layers.split(','))
    model = EfficientNet_CBAM(len(classes), pretrained=not a.no_pretrained, train_cbam=a.train_cbam,
                              finetune_layers=layers).to(dev)
    opt = torch.optim.AdamW(model.param_groups(a.lr), weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total params: {sum(p.numel() for p in model.parameters()):,}   trainable: {n_train:,}   '
          f'unfrozen blocks: {layers}{" + cbam" if a.train_cbam else ""}')

    best_val, best_epoch, no_improve = -1.0, -1, 0
    for ep in range(a.epochs):
        model.train(); tot, corr, n = 0.0, 0, 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); out = model(x); loss = crit(out, y); loss.backward(); opt.step()
            tot += loss.item() * len(y); corr += (out.argmax(1) == y).sum().item(); n += len(y)
        sched.step()
        val_acc = evaluate(model, va, dev)
        if val_acc > best_val:
            best_val, best_epoch, no_improve = val_acc, ep + 1, 0
            torch.save(model.state_dict(), a.out)
        else:
            no_improve += 1
        print(f'Epoch {ep + 1}/{a.epochs} - loss {tot / n:.4f} - train acc {corr / n:.4f} - val acc {val_acc:.4f}'
              f'{"  <- best" if no_improve == 0 else ""}', flush=True)
        if no_improve >= a.patience:
            print(f'early stopping: no val improvement for {a.patience} epochs (best was epoch {best_epoch})')
            break
    torch.save(model.state_dict(), a.out.replace('.pth', '_last.pth'))

    # ---- final, ONE-TIME evaluation on the held-out test set, using the best (early-stopped) checkpoint.
    # The test set was never used to pick an epoch or tune anything, so this is the honest number to report.
    model.load_state_dict(torch.load(a.out, map_location=dev))
    from sklearn.metrics import classification_report
    test_acc, yt, yp = evaluate(model, te, dev, return_preds=True, tta=a.tta)
    print(f'\nbest-epoch VAL accuracy (epoch {best_epoch}):                {best_val:.4f}')
    print(f'held-out TEST accuracy (never seen during training{", TTA" if a.tta else ""}): {test_acc:.4f}')
    if yt:
        print('\nTest set classification report:')
        print(classification_report(yt, yp, labels=list(range(len(classes))), target_names=classes, digits=4, zero_division=0))

if __name__ == '__main__':
    main()
