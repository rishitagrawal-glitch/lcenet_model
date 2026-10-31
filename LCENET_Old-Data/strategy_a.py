"""
strategy_a.py  -  LCE-Net (4.02M params)  ->  < 50,000 params

  Phase 1  LAMP coarse pruning         (global structured pruning, LAMP scores, no layer collapse)   ~4.0M -> ~1.0M
  Phase 2  Architectural surgery       (cut the 1280-ch head + late stages, re-attach a small head)  ~1.0M -> <=400k
  Phase 3  Iterative Taylor pruning    (data-driven, many small steps, KD fine-tune between steps)   ->  < 50k
  Phase 4  Knowledge-distillation fine-tune (teacher = the ORIGINAL, unpruned LCE-Net)

Needs the trained baseline checkpoint made by baseline_lcenet.py (or the repo's effnet.py).

Test accuracy: uses the same train/val/test split as baseline_lcenet.py (default 70/15/15, leak-free).
VAL is used throughout pruning/fine-tuning to print progress after each phase (so you can see it recovering
step by step). TEST is never touched until the very end - it's printed once, after phase 4, as the number
to actually report. Both are printed for every phase so you can compare.

  pip install torch torchvision torch-pruning scikit-learn
  python baseline_lcenet.py --data ./dataset/Potato --epochs 50            # -> best_model_cbam.pth
  python strategy_a.py      --data ./dataset/Potato --ckpt best_model_cbam.pth --split group
"""
import argparse, copy, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch_pruning as tp
from baseline_lcenet import EfficientNet_CBAM, Lite_CBAM, make_loaders, evaluate


def n_params(m): return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------- model wrappers
class FullNet(nn.Module):
    """Copy of the teacher without the duplicate `head` alias, so the pruner sees each module once."""
    def __init__(self, t):
        super().__init__()
        self.features, self.cbam = copy.deepcopy(t.features), copy.deepcopy(t.cbam)
        self.pool, self.dropout, self.classifier = nn.AdaptiveAvgPool2d(1), nn.Dropout(0.2), copy.deepcopy(t.classifier)

    def forward(self, x):
        for i in range(8):
            x = self.features[i](x)
        x = self.features[8](self.cbam(x))
        return self.classifier(self.dropout(torch.flatten(self.pool(x), 1)))


class Truncated(nn.Module):
    """Phase-2 surgery: keep features[0..stage], DELETE the rest (incl. the 320->1280 expansion), then
    fresh Lite-CBAM -> global pool -> Linear(3)."""
    def __init__(self, full, stage, num_classes):
        super().__init__()
        self.features = nn.Sequential(*[copy.deepcopy(full.features[i]) for i in range(stage + 1)])
        c = [m for m in self.features[-1].modules() if isinstance(m, nn.Conv2d)][-1].out_channels
        self.cbam, self.pool, self.classifier = Lite_CBAM(c, reduction=4), nn.AdaptiveAvgPool2d(1), nn.Linear(c, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.cbam(self.features(x))), 1))


# ----------------------------------------------------------------------------- training helpers
def kd_loss(s, t, y, T=4.0, a=0.7):
    return a * T * T * F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction='batchmean') \
        + (1 - a) * F.cross_entropy(s, y, label_smoothing=0.05)


def finetune(student, teacher, tr, va, dev, epochs, lr, tag):
    hist = []
    if epochs <= 0:
        return hist
    opt = torch.optim.AdamW(student.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(tr))
    teacher.eval()
    for ep in range(epochs):
        student.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            with torch.no_grad():
                t = teacher(x)
            loss = kd_loss(student(x), t, y); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append(evaluate(student, va, dev))
        print(f'    [{tag}] epoch {ep + 1}/{epochs} loss={loss.item():.3f} val_acc={hist[-1]:.4f}', flush=True)
    return hist


def report(name, student, va, te, dev, rows):
    """Print + record BOTH val and test accuracy for a phase checkpoint."""
    v, t = evaluate(student, va, dev), evaluate(student, te, dev)
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./dataset/Potato'); ap.add_argument('--ckpt', default='best_model_cbam.pth')
    ap.add_argument('--split', choices=['group', 'repo'], default='group')
    ap.add_argument('--res', type=int, default=224); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--budget', type=int, default=49000, help='final param budget (must be < 50,000)')
    ap.add_argument('--phase1_keep', type=float, default=0.25, help='fraction of params kept after LAMP (0.25 = -75%%)')
    ap.add_argument('--phase2_cap', type=int, default=400_000, help='pick deepest stage with <= this many params')
    ap.add_argument('--taylor_steps', type=int, default=10); ap.add_argument('--taylor_batches', type=int, default=8)
    ap.add_argument('--ft1', type=int, default=3, help='recovery epochs after phase 1')
    ap.add_argument('--ft2', type=int, default=3, help='recovery epochs after phase 2')
    ap.add_argument('--ft_step', type=int, default=1, help='epochs after each Taylor step')
    ap.add_argument('--ft_final', type=int, default=30, help='phase-4 KD epochs')
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--out', default='lcenet_pruned')
    ap.add_argument('--limit', type=int, default=None, help='debug: use only N images')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(0)
    tr, va, te, classes = make_loaders(a.data, a.split, a.res, a.bs, limit=a.limit)
    ex = torch.randn(1, 3, a.res, a.res, device=dev)
    rows = []                                            # (phase name, params, val_acc, test_acc)

    # ---- teacher (original, frozen, unpruned)
    teacher = EfficientNet_CBAM(len(classes), pretrained=not os.path.exists(a.ckpt))
    if os.path.exists(a.ckpt):
        teacher.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
    else:
        print('!! checkpoint not found: using ImageNet weights + untrained head. Run baseline_lcenet.py first.')
    teacher = teacher.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print('[baseline]'); report('0. Baseline LCE-Net (teacher)', teacher, va, te, dev, rows)

    student = FullNet(teacher).to(dev)
    for p in student.parameters():
        p.requires_grad_(True)                          # baseline froze most weights; the student must train

    # ================= Phase 1: LAMP coarse pruning =================
    print('\n=== Phase 1: LAMP coarse pruning (global, structured) ===')
    target1 = int(a.phase1_keep * n_params(student))
    lamp = lambda: tp.importance.LAMPImportance()
    r1 = search_ratio(student, ex, target1, lamp, global_pruning=True, max_ratio=0.9)
    make_pruner(student, ex, lamp(), r1, 1, True, 0.9).step()
    print(f'  global ratio {r1:.3f}: {n_params(teacher):,} -> {n_params(student):,} params')
    finetune(student, teacher, tr, va, dev, a.ft1, a.lr / 3, 'phase1')
    report('1. LAMP coarse prune', student, va, te, dev, rows)

    # ================= Phase 2: architectural surgery =================
    print('\n=== Phase 2: architectural surgery (delete 1280-ch head + late stages) ===')
    chosen = None
    for k in range(2, 8):
        p = n_params(Truncated(student, k, len(classes)))
        print(f'  cut after features[{k}]: {p:,} params')
        if p <= a.phase2_cap:
            chosen = k
    chosen = chosen if chosen is not None else 2
    student = Truncated(student, chosen, len(classes)).to(dev)
    print(f'  -> chosen stage {chosen}: {n_params(student):,} params')
    finetune(student, teacher, tr, va, dev, a.ft2, a.lr, 'phase2')
    report(f'2. surgery (stage {chosen})', student, va, te, dev, rows)

    # ================= Phase 3: iterative Taylor pruning =================
    print('\n=== Phase 3: iterative first-order Taylor pruning ===')
    r3 = search_ratio(student, ex, a.budget, lambda: tp.importance.GroupMagnitudeImportance(p=2), global_pruning=False)
    print(f'  channel ratio {r3:.3f} to reach <= {a.budget:,} params, in {a.taylor_steps} steps')
    pruner = make_pruner(student, ex, tp.importance.GroupTaylorImportance(), r3, a.taylor_steps, False)
    for s in range(a.taylor_steps):
        student.train(); student.zero_grad()
        for b, (x, y) in enumerate(tr):                 # accumulate |w * dL/dw| on real potato batches
            if b >= a.taylor_batches:
                break
            x, y = x.to(dev), y.to(dev)
            with torch.no_grad():
                t = teacher(x)
            kd_loss(student(x), t, y).backward()
        pruner.step(); student.zero_grad()
        print(f'  step {s + 1}/{a.taylor_steps}: {n_params(student):,} params')
        finetune(student, teacher, tr, va, dev, a.ft_step, a.lr / 3, f'taylor{s + 1}')
    report('3. Taylor prune', student, va, te, dev, rows)

    # ================= Phase 4: KD fine-tune =================
    print('\n=== Phase 4: knowledge-distillation fine-tune (teacher = original LCE-Net) ===')
    hist = finetune(student, teacher, tr, va, dev, a.ft_final, a.lr, 'phase4')
    final_val, final_test = report('4. KD fine-tune (FINAL)', student, va, te, dev, rows)

    from sklearn.metrics import classification_report
    _, yt, yp = evaluate(student, te, dev, return_preds=True)
    print('\nFinal model - TEST SET classification report (held out the entire run):')
    print(classification_report(yt, yp, labels=list(range(len(classes))), target_names=classes, digits=4, zero_division=0))

    print(f'\n{"phase":<28}{"params":>12}{"val acc":>10}{"test acc":>10}')
    for name, p, v, t in rows:
        print(f'{name:<28}{p:>12,}{v:>10.4f}{t:>10.4f}')
    if hist:
        print(f'\nlast-5-epoch mean VAL acc (phase 4): {np.mean(hist[-5:]):.4f}')
    print(f'FINAL TEST accuracy (report this one): {final_test:.4f}')

    assert n_params(student) < 50_000, f'{n_params(student):,} params - budget missed; lower --budget'
    student = student.cpu().eval()
    torch.jit.trace(student, torch.randn(1, 3, a.res, a.res)).save(a.out + '.pt')
    print(f'\nOK: {n_params(student):,} params (< 50,000). Saved {a.out}.pt  (torch.jit.load; input normalised mean=std=0.5)')

if __name__ == '__main__':
    main()
