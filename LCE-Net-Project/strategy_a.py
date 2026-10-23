"""
strategy_a.py  -  LCE-Net (4.02M params)  ->  ~90,000 params (configurable budget)

  Phase 1  LAMP coarse pruning         (global structured pruning, LAMP scores, no layer collapse)   ~4.0M -> ~1.0M
  Phase 2  Architectural surgery       (cut the 1280-ch head + late stages, re-attach head for 7 classes) ~1.0M -> <=400k
  Phase 3  Iterative Taylor pruning    (data-driven, many small steps, KD fine-tune between steps)   ->  ~90k budget
  Phase 4  Knowledge-distillation fine-tune (teacher = the ORIGINAL, unpruned LCE-Net from final_lcenet_new.py)

Prunes the improved LCE-Net model from final_lcenet_new.py (supporting 7 disease classes from dataset_extracted).

Test accuracy: uses the same train/val/test split and ImageNet normalization as final_lcenet_new.py (default 70/15/15, leak-free).
VAL is used throughout pruning/fine-tuning to print progress after each phase (so you can see it recovering
step by step). TEST is held out until the very end and reported once after phase 4 as the honest benchmark.

Usage:
  # 1. Train final_lcenet_new (creates best_model_final.pth with 7 classes)
  python final_lcenet_new.py --data ./dataset_extracted --epochs 60 --oversample

  # 2. Prune it down to ~90,000 parameters using Strategy A
  python strategy_a.py   --data ./dataset_extracted --ckpt best_model_final.pth --budget 90000 --split group
"""
import argparse, copy, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
import torch_pruning as tp
from final_lcenet_new import EfficientNet_CBAM, Lite_CBAM, FocalLoss, three_way_split


def n_params(m): return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------- data loaders
def make_loaders(root, split='group', res=224, bs=32, seed=42, oversample=False, limit=None, workers=0, strong_aug=False):
    """
    Clean data loader that restores the proven, gentle augmentations from the 79% run:
      - Resize(224, 224) -> no cropping out disease spots
      - RandomHorizontalFlip
      - RandomRotation(15)
      - ColorJitter(0.2, 0.2, 0.2)
      - Normalization: ImageNet stats matching best_model_final.pth
      - NO RandomErasing (which deletes disease lesions)
      - NO GaussianBlur (which blurs fine fungal/bacterial spots)
      - NO RandomGrayscale (which destroys leaf color diagnosis)
    """
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if strong_aug:
        tf_train = transforms.Compose([
            transforms.Resize((res + 32, res + 32)),
            transforms.RandomCrop(res),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.GaussianBlur(3, (0.1, 2.0)),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:
        tf_train = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            norm,
        ])
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
        va = va[:max(1, limit // 4)]
        te = te[:max(1, limit // 4)]
    print(f'split={split}: train={len(tr)} val={len(va)} test={len(te)} classes={ds_tr.classes}')
    labels = [ds_tr.samples[i][1] for i in tr]
    if oversample:
        counts = np.bincount(labels)
        w = 1.0 / counts
        sampler = WeightedRandomSampler([w[l] for l in labels], len(labels), True)
        train_loader = DataLoader(Subset(ds_tr, tr), bs, sampler=sampler, num_workers=workers, pin_memory=True)
    else:
        train_loader = DataLoader(Subset(ds_tr, tr), bs, shuffle=True, num_workers=workers, pin_memory=True, drop_last=len(tr) > bs)
    val_loader = DataLoader(Subset(ds_ev, va), bs, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(Subset(ds_ev, te), bs, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, test_loader, ds_tr.classes


@torch.no_grad()
def evaluate(model, loader, dev, return_preds=False, tta=False):
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        x = x.to(dev)
        logits = model(x)
        if tta:
            # Test-time augmentation: average original and horizontally-flipped prediction
            logits = (logits + model(torch.flip(x, dims=[3]))) / 2.0
        yp += logits.argmax(1).cpu().tolist()
        yt += y.tolist()
    acc = float(np.mean(np.array(yt) == np.array(yp))) if yt else float('nan')
    return (acc, yt, yp) if return_preds else acc


# ----------------------------------------------------------------------------- model wrappers
class FullNet(nn.Module):
    """Copy of the teacher without the duplicate `head` alias, so the pruner sees each module once."""
    def __init__(self, t):
        super().__init__()
        self.features, self.cbam = copy.deepcopy(t.features), copy.deepcopy(t.cbam)
        self.pool, self.dropout, self.classifier = (
            nn.AdaptiveAvgPool2d(1),
            copy.deepcopy(t.dropout),
            copy.deepcopy(t.classifier)
        )

    def forward(self, x):
        for i in range(8):
            x = self.features[i](x)
        x = self.features[8](self.cbam(x))
        return self.classifier(self.dropout(torch.flatten(self.pool(x), 1)))


class Truncated(nn.Module):
    """Phase-2 surgery: keep features[0..stage], DELETE the rest (incl. the 320->1280 expansion), then
    fresh Lite-CBAM -> global pool -> Linear(num_classes)."""
    def __init__(self, full, stage, num_classes):
        super().__init__()
        self.features = nn.Sequential(*[copy.deepcopy(full.features[i]) for i in range(stage + 1)])
        c = [m for m in self.features[-1].modules() if isinstance(m, nn.Conv2d)][-1].out_channels
        self.cbam, self.pool, self.classifier = Lite_CBAM(c, reduction=4), nn.AdaptiveAvgPool2d(1), nn.Linear(c, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.cbam(self.features(x))), 1))


# ----------------------------------------------------------------------------- training helpers
def kd_loss(s, t, y, criterion=None, T=4.0, a=0.7):
    soft_loss = a * T * T * F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction='batchmean')
    if criterion is not None:
        hard_loss = (1 - a) * criterion(s, y)
    else:
        hard_loss = (1 - a) * F.cross_entropy(s, y, label_smoothing=0.05)
    return soft_loss + hard_loss


def finetune(student, teacher, tr, va, dev, epochs, lr, tag, criterion=None, save_best=False):
    hist = []
    if epochs <= 0:
        return hist
    opt = torch.optim.AdamW(student.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(tr))
    teacher.eval()
    best_state, best_acc = None, -1.0
    for ep in range(epochs):
        student.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            with torch.no_grad():
                t = teacher(x)
            loss = kd_loss(student(x), t, y, criterion=criterion)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        val_acc = evaluate(student, va, dev)
        hist.append(val_acc)
        if save_best and val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(student.state_dict())
            print(f'    [{tag}] epoch {ep + 1}/{epochs} loss={loss.item():.3f} val_acc={val_acc:.4f} <- best', flush=True)
        else:
            print(f'    [{tag}] epoch {ep + 1}/{epochs} loss={loss.item():.3f} val_acc={val_acc:.4f}', flush=True)
    if save_best and best_state is not None:
        student.load_state_dict(best_state)
        print(f'    [{tag}] Reloaded best checkpoint from training (val_acc={best_acc:.4f})', flush=True)
    return hist


def report(name, student, va, te, dev, rows, tta=False):
    """Print + record BOTH val and test accuracy for a phase checkpoint."""
    v = evaluate(student, va, dev)
    t = evaluate(student, te, dev, tta=tta)
    rows.append((name, n_params(student), v, t))
    print(f'  -> {name}: params={n_params(student):,}  val_acc={v:.4f}  test_acc={t:.4f}')
    return v, t


# ----------------------------------------------------------------------------- pruning helpers
def make_pruner(model, ex, importance, ratio, steps=1, global_pruning=False, max_ratio=1.0):
    return tp.pruner.BasePruner(model, ex, importance=importance, pruning_ratio=ratio, iterative_steps=steps,
                                global_pruning=global_pruning, max_pruning_ratio=max_ratio,
                                ignored_layers=[model.classifier], round_to=None)


def search_ratio(model, ex, target, importance_fn, global_pruning, max_ratio=1.0, iters=12):
    """Bisection: smallest pruning ratio whose one-shot result has <= `target` params."""
    lo, hi = 0.0, 0.98
    for _ in range(iters):
        mid = (lo + hi) / 2; m = copy.deepcopy(model).cpu()
        make_pruner(m, ex.cpu(), importance_fn(), mid, 1, global_pruning, max_ratio).step()
        lo, hi = (lo, mid) if n_params(m) <= target else (mid, hi)
    return hi


def main():
    ap = argparse.ArgumentParser(description='Strategy A: 3-phase Pruning on final_lcenet')
    default_data = './dataset_extracted' if os.path.exists('./dataset_extracted') else './dataset/Potato'
    default_ckpt = 'best_model_final.pth'
    ap.add_argument('--data', default=default_data, help=f'dataset path (default: {default_data})')
    ap.add_argument('--ckpt', default=default_ckpt, help=f'checkpoint path (default: {default_ckpt})')
    ap.add_argument('--split', choices=['group', 'repo'], default='group')
    ap.add_argument('--res', type=int, default=224)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--budget', type=int, default=90000, help='final param budget (default: 90000 for ~90k params)')
    ap.add_argument('--phase1_keep', type=float, default=0.25, help='fraction of params kept after LAMP (0.25 = -75%%)')
    ap.add_argument('--phase2_cap', type=int, default=400_000, help='pick deepest stage with <= this many params')
    ap.add_argument('--taylor_steps', type=int, default=10)
    ap.add_argument('--taylor_batches', type=int, default=8, help='batches for Taylor gradient sensitivity')
    ap.add_argument('--ft1', type=int, default=3, help='recovery epochs after phase 1')
    ap.add_argument('--ft2', type=int, default=3, help='recovery epochs after phase 2')
    ap.add_argument('--ft_step', type=int, default=1, help='epochs after each Taylor step')
    ap.add_argument('--ft_final', type=int, default=30, help='phase-4 KD epochs')
    ap.add_argument('--lr', type=float, default=3e-3, help='learning rate for KD fine-tuning')
    ap.add_argument('--oversample', action='store_true', default=False, help='balance classes via WeightedRandomSampler')
    ap.add_argument('--strong_aug', action='store_true', default=False, help='use aggressive augmentations (RandomErasing/GaussianBlur)')
    ap.add_argument('--focal_gamma', type=float, default=0.0, help='Focal loss gamma (0 = clean CrossEntropy from 79pct baseline)')
    ap.add_argument('--weighted_loss', action='store_true', default=False, help='apply inverse-frequency class weights')
    ap.add_argument('--tta', action='store_true', default=True, help='use Test-Time Augmentation (flip averaging)')
    ap.add_argument('--no_tta', dest='tta', action='store_false')
    ap.add_argument('--workers', type=int, default=0, help='DataLoader workers (default 0 for Windows)')
    ap.add_argument('--out', default='lcenet_pruned')
    ap.add_argument('--limit', type=int, default=None, help='debug: use only N images')
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    tr, va, te, classes = make_loaders(
        a.data, a.split, a.res, a.bs,
        oversample=a.oversample, limit=a.limit, workers=a.workers, strong_aug=a.strong_aug
    )
    num_classes = len(classes)
    ex = torch.randn(1, 3, a.res, a.res, device=dev)
    rows = []                                            # (phase name, params, val_acc, test_acc)

    # ---- Loss criterion (defaults to clean standard CrossEntropy from the 79% baseline)
    if a.weighted_loss:
        train_labels = [tr.dataset.dataset.samples[i][1] for i in tr.dataset.indices]
        counts = np.bincount(train_labels, minlength=num_classes).astype(float)
        counts = np.maximum(counts, 1.0)
        class_weights = torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32).to(dev)
    else:
        class_weights = None

    if a.focal_gamma > 0:
        criterion = FocalLoss(gamma=a.focal_gamma, weight=class_weights, label_smoothing=0.05)
    elif class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    else:
        criterion = None  # None -> uses clean F.cross_entropy(s, y, label_smoothing=0.05) exactly as in the 79% baseline

    # ---- Teacher (original, frozen, unpruned from final_lcenet)
    teacher = EfficientNet_CBAM(num_classes=num_classes, pretrained=not os.path.exists(a.ckpt))
    if os.path.exists(a.ckpt):
        print(f'Loading teacher weights from {a.ckpt}...')
        ckpt_state = torch.load(a.ckpt, map_location='cpu')
        if 'classifier.weight' in ckpt_state and ckpt_state['classifier.weight'].shape[0] != num_classes:
            print(f'WARNING: Checkpoint classifier has {ckpt_state["classifier.weight"].shape[0]} classes, '
                  f'but dataset has {num_classes} classes ({classes}). '
                  f'Loading backbone/CBAM weights and reinitializing classifier.')
            ckpt_state = {k: v for k, v in ckpt_state.items() if not k.startswith('classifier')}
            teacher.load_state_dict(ckpt_state, strict=False)
        else:
            teacher.load_state_dict(ckpt_state)
    else:
        print(f'!! Checkpoint "{a.ckpt}" not found: using ImageNet weights + untrained head.')
        print('   For best results, train the model first with final_lcenet.py to produce best_model_final.pth.')

    teacher = teacher.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print('[teacher]'); report('0. Baseline LCE-Net (teacher)', teacher, va, te, dev, rows)

    student = FullNet(teacher).to(dev)
    for p in student.parameters():
        p.requires_grad_(True)

    # ================= Phase 1: LAMP coarse pruning =================
    print('\n=== Phase 1: LAMP coarse pruning (global, structured) ===')
    target1 = int(a.phase1_keep * n_params(student))
    lamp = lambda: tp.importance.LAMPImportance()
    r1 = search_ratio(student, ex, target1, lamp, global_pruning=True, max_ratio=0.9)
    make_pruner(student, ex, lamp(), r1, 1, True, 0.9).step()
    print(f'  global ratio {r1:.3f}: {n_params(teacher):,} -> {n_params(student):,} params')
    finetune(student, teacher, tr, va, dev, a.ft1, a.lr / 3, 'phase1', criterion=criterion)
    report('1. LAMP coarse prune', student, va, te, dev, rows)

    # ================= Phase 2: architectural surgery =================
    print('\n=== Phase 2: architectural surgery (delete 1280-ch head + late stages) ===')
    chosen = None
    for k in range(2, 8):
        p = n_params(Truncated(student, k, num_classes))
        print(f'  cut after features[{k}]: {p:,} params')
        if a.budget <= p <= a.phase2_cap:
            chosen = k
        elif chosen is None and p >= a.budget:
            chosen = k
    chosen = chosen if chosen is not None else 5
    student = Truncated(student, chosen, num_classes).to(dev)
    print(f'  -> chosen stage {chosen}: {n_params(student):,} params (attached head for {num_classes} classes)')
    finetune(student, teacher, tr, va, dev, a.ft2, a.lr, 'phase2', criterion=criterion)
    report(f'2. surgery (stage {chosen})', student, va, te, dev, rows)

    # ================= Phase 3: iterative Taylor pruning =================
    print('\n=== Phase 3: iterative first-order Taylor pruning ===')
    r3 = search_ratio(student, ex, a.budget, lambda: tp.importance.GroupMagnitudeImportance(p=2), global_pruning=False)
    print(f'  channel ratio {r3:.3f} to reach <= {a.budget:,} params, in {a.taylor_steps} steps')
    pruner = make_pruner(student, ex, tp.importance.GroupTaylorImportance(), r3, a.taylor_steps, False)
    for s in range(a.taylor_steps):
        student.train(); student.zero_grad()
        for b, (x, y) in enumerate(tr):                 # accumulate |w * dL/dw| on real batches
            if b >= a.taylor_batches:
                break
            x, y = x.to(dev), y.to(dev)
            with torch.no_grad():
                t = teacher(x)
            kd_loss(student(x), t, y, criterion=criterion).backward()
        pruner.step(); student.zero_grad()
        print(f'  step {s + 1}/{a.taylor_steps}: {n_params(student):,} params')
        finetune(student, teacher, tr, va, dev, a.ft_step, a.lr / 3, f'taylor{s + 1}', criterion=criterion)
    report('3. Taylor prune', student, va, te, dev, rows)

    # ================= Phase 4: KD fine-tune =================
    print('\n=== Phase 4: knowledge-distillation fine-tune (teacher = original final_lcenet) ===')
    hist = finetune(student, teacher, tr, va, dev, a.ft_final, a.lr, 'phase4', criterion=criterion, save_best=True)
    report('4. KD fine-tune', student, va, te, dev, rows, tta=a.tta)

    from sklearn.metrics import classification_report, confusion_matrix
    final_test, yt, yp = evaluate(student, te, dev, return_preds=True, tta=a.tta)
    print(f'\nFinal model - TEST SET classification report (held out the entire run{", TTA enabled" if a.tta else ""}):')
    print(classification_report(yt, yp, labels=list(range(num_classes)), target_names=classes, digits=4, zero_division=0))

    if yt:
        cm = confusion_matrix(yt, yp, labels=list(range(num_classes)))
        print('Per-class accuracy:')
        for i, cls in enumerate(classes):
            cls_total = cm[i].sum()
            cls_correct = cm[i, i]
            acc_str = f'{cls_correct / cls_total:.4f}' if cls_total > 0 else 'N/A'
            print(f'  {cls:<15} {cls_correct}/{cls_total} = {acc_str}')

    print(f'\n{"phase":<28}{"params":>12}{"val acc":>10}{"test acc":>10}')
    for name, p, v, t in rows:
        print(f'{name:<28}{p:>12,}{v:>10.4f}{t:>10.4f}')
    if hist:
        print(f'\nlast-5-epoch mean VAL acc (phase 4): {np.mean(hist[-5:]):.4f}')
    print(f'FINAL TEST accuracy (report this one{", TTA" if a.tta else ""}): {final_test:.4f}')

    assert n_params(student) <= a.budget, f'{n_params(student):,} params - budget missed (target <= {a.budget:,}); adjust --budget'
    student = student.cpu().eval()
    torch.jit.trace(student, torch.randn(1, 3, a.res, a.res)).save(a.out + '.pt')
    torch.save(student.state_dict(), a.out + '.pth')
    print(f'\nOK: {n_params(student):,} params (target budget <= {a.budget:,}, {num_classes} classes). Saved {a.out}.pt and {a.out}.pth (input normalised with ImageNet mean/std)')


if __name__ == '__main__':
    main()
