"""
baseline_lcenet.py  -  ONLY the baseline model: LCE-Net = EfficientNet-B0 + Lite-CBAM (moufamiao/LCE-Net).

Model, freezing scheme, optimiser and learning rates are identical to the repo's effnet.py, so
`best_model_cbam.pth` from this script is interchangeable with the repo's checkpoint.

Three-way split (this is the main change from the original repo, which only has train/val):
  * TRAIN  used to update weights.
  * VAL    used only to pick the best epoch to checkpoint (model selection). Seen many times during training.
  * TEST   held out completely during training and epoch selection. Reported ONCE at the very end, after
           reloading the best checkpoint. This is the honest "test accuracy" number.
  Default split is 70/15/15. The original repo has no test set at all - it reports val accuracy and calls
  it done, which is why its 100% is easy to get: that same val set decided which checkpoint got saved.

Other differences from the repo (all optional / bug fixes):
  * --split group   leak-free split (augmented siblings of a healthy photo stay together). Default.
    --split repo    same 80/20 proportions as the repo's random_split(seed=42), just extended to 3-way.
  * train and val/test use SEPARATE dataset objects. In the repo all subsets share one dataset object, so
    the eval transform overwrites the train augmentation and augmentation never actually runs.
  * --train_cbam    also trains the CBAM. In the repo it is frozen at RANDOM init (its parameter names contain
    neither "features.6" nor "classifier"), so the attention is never learned. Off by default = faithful baseline.

Usage:  python baseline_lcenet.py --data ./dataset/Potato --epochs 50
"""
import argparse, os, re
import numpy as np
import torch, torch.nn as nn
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
    def __init__(self, num_classes=3, pretrained=True, train_cbam=False):
        super().__init__()
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_b0(weights=w)
        self.features = base.features
        self.cbam = Lite_CBAM(channels=320)
        self.head = self.features[8]                     # Conv2d(320 -> 1280); alias kept for checkpoint parity
        self.pool = base.avgpool
        self.dropout = base.classifier[0]
        self.classifier = nn.Linear(1280, num_classes)
        for p in self.parameters():
            p.requires_grad = False
        for name, p in self.named_parameters():
            if "features.6" in name or "classifier" in name or (train_cbam and name.startswith("cbam")):
                p.requires_grad = True

    def forward(self, x):
        for i in range(8):
            x = self.features[i](x)
        x = self.cbam(x)
        x = self.head(x)
        x = torch.flatten(self.pool(x), 1)
        return self.classifier(self.dropout(x))


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
                                   transforms.RandomRotation(15), transforms.ColorJitter(0.2, 0.2, 0.2),
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
def evaluate(model, loader, dev, return_preds=False):
    model.eval(); yt, yp = [], []
    for x, y in loader:
        yp += model(x.to(dev)).argmax(1).cpu().tolist(); yt += y.tolist()
    acc = float(np.mean(np.array(yt) == np.array(yp))) if yt else float('nan')
    return (acc, yt, yp) if return_preds else acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./dataset/Potato'); ap.add_argument('--split', choices=['group', 'repo'], default='group')
    ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--res', type=int, default=224); ap.add_argument('--oversample', action='store_true')
    ap.add_argument('--train_cbam', action='store_true'); ap.add_argument('--no_pretrained', action='store_true')
    ap.add_argument('--out', default='best_model_cbam.pth'); ap.add_argument('--limit', type=int, default=None, help='debug: N images')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr, va, te, classes = make_loaders(a.data, a.split, a.res, a.bs, oversample=a.oversample, limit=a.limit)
    model = EfficientNet_CBAM(len(classes), pretrained=not a.no_pretrained, train_cbam=a.train_cbam).to(dev)
    groups = [{'params': model.features[6].parameters(), 'lr': 1e-5},
              {'params': model.classifier.parameters(), 'lr': 5e-4}]
    if a.train_cbam:
        groups.append({'params': model.cbam.parameters(), 'lr': 5e-4})
    opt = torch.optim.Adam(groups); crit = nn.CrossEntropyLoss()
    print(f'total params: {sum(p.numel() for p in model.parameters()):,}   '
          f'trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    best_val = -1.0
    for ep in range(a.epochs):
        model.train(); tot, corr, n = 0.0, 0, 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); out = model(x); loss = crit(out, y); loss.backward(); opt.step()
            tot += loss.item() * len(y); corr += (out.argmax(1) == y).sum().item(); n += len(y)
        val_acc = evaluate(model, va, dev)
        if val_acc > best_val:
            best_val = val_acc; torch.save(model.state_dict(), a.out)
        print(f'Epoch {ep + 1}/{a.epochs} - loss {tot / n:.4f} - train acc {corr / n:.4f} - val acc {val_acc:.4f}', flush=True)
    torch.save(model.state_dict(), a.out.replace('.pth', '_last.pth'))

    # ---- final, ONE-TIME evaluation on the held-out test set, using the best (early-stopped) checkpoint.
    # The test set was never used to pick an epoch or tune anything, so this is the honest number to report.
    model.load_state_dict(torch.load(a.out, map_location=dev))
    from sklearn.metrics import classification_report
    test_acc, yt, yp = evaluate(model, te, dev, return_preds=True)
    print(f'\nbest-epoch VAL accuracy (used for checkpoint selection): {best_val:.4f}')
    print(f'held-out TEST accuracy (never seen during training):     {test_acc:.4f}')
    if yt:
        print('\nTest set classification report:')
        print(classification_report(yt, yp, labels=list(range(len(classes))), target_names=classes, digits=4, zero_division=0))

if __name__ == '__main__':
    main()
