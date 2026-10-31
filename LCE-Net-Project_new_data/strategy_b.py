"""
strategy_b.py  -  LCE-Net (4.02M params) -> < 50,000 params, in ONE unified pruning pass.

This is the "Taylor-LAMP Hybrid" strategy: instead of running LAMP, then manual surgery, then Taylor as
three separate steps (strategy_a.py), it scores every filter in the whole network with a single formula
that is both DATA-AWARE (Taylor) and LAYER-SAFE (LAMP), then makes ONE global cut straight to the budget.

Step 1 - Taylor score (data-aware): for every filter/channel, run a few real batches through the network,
         backprop, and compute |W * dL/dW| - how much the loss would change if this filter were deleted.
         A filter with small weights can still score high here if removing it hurts the model.
Step 2 - LAMP normalization (layer-safe): WITHIN each layer, sort those Taylor scores and rescale them
         with LAMP's cumulative-sum formula:
             hybrid_score(i) = taylor_score(i)^2 / sum_{j >= i} taylor_score(j)^2   (scores sorted ascending)
         This guarantees the single most-important filter in EVERY layer gets score 1.0, so no layer can
         ever be reduced to zero filters - the layer-collapse failure mode of plain Taylor pruning.
Step 3 - Global cut: pool every filter's hybrid score from every layer into one list, sort it, and prune
         the lowest-scoring filters across the WHOLE network (not layer-by-layer) until the parameter
         budget is met. Because of step 2, this includes the oversized 320->1280 head automatically - no
         manual "delete this layer" surgery step is needed the way strategy_a.py needs one.

This is implemented as a single custom importance class (~10 lines) that subclasses torch-pruning's own
GroupTaylorImportance and forces its LAMP normalizer on - torch-pruning already ships both halves of the
formula, this just combines them.

  pip install torch torchvision torch-pruning scikit-learn
  python baseline_lcenet.py --data ./dataset/Potato --epochs 60                       # -> best_model_cbam.pth
  python strategy_b.py      --data ./dataset/Potato --ckpt best_model_cbam.pth --split group
"""
import argparse, copy, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch_pruning as tp
from baseline_lcenet import EfficientNet_CBAM, make_loaders, evaluate


def n_params(m): return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------- THE hybrid importance
class TaylorLAMPImportance(tp.importance.GroupTaylorImportance):
    """Taylor's data-aware |W * dL/dW| score, LAMP-normalized within each layer so every layer keeps at
    least one filter with score 1.0. See module docstring, Step 1 + Step 2."""
    def __init__(self, group_reduction='mean', multivariable=False, bias=False):
        super().__init__(group_reduction=group_reduction, normalizer='lamp', multivariable=multivariable, bias=bias)


# ----------------------------------------------------------------------------- model wrapper (dedupe head alias)
class FullNet(nn.Module):
    """Copy of the teacher without the duplicate `head` alias (features[8] appears twice under two names
    in EfficientNet_CBAM), so the pruner's dependency graph sees each module exactly once."""
    def __init__(self, t):
        super().__init__()
        self.features, self.cbam = copy.deepcopy(t.features), copy.deepcopy(t.cbam)
        self.pool, self.dropout, self.classifier = nn.AdaptiveAvgPool2d(1), nn.Dropout(0.2), copy.deepcopy(t.classifier)

    def forward(self, x):
        for i in range(8):
            x = self.features[i](x)
        x = self.features[8](self.cbam(x))
        return self.classifier(self.dropout(torch.flatten(self.pool(x), 1)))


# ----------------------------------------------------------------------------- training / eval helpers
def kd_loss(s, t, y, T=4.0, a=0.7):
    return a * T * T * F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction='batchmean') \
        + (1 - a) * F.cross_entropy(s, y, label_smoothing=0.05)


def accumulate_gradients(student, teacher, tr, dev, n_batches):
    """Step 1 of the hybrid score: real potato-leaf batches through the (currently unpruned) student,
    accumulating dL/dW in .grad via repeated .backward() calls (no zero_grad between them - this is what
    GroupTaylorImportance reads)."""
    student.train(); student.zero_grad()
    for b, (x, y) in enumerate(tr):
        if b >= n_batches:
            break
        x, y = x.to(dev), y.to(dev)
        with torch.no_grad():
            t = teacher(x)
        kd_loss(student(x), t, y).backward()


def deepcopy_with_grad(model):
    """copy.deepcopy() drops .grad on parameters (verified: Tensor.__deepcopy__ does not clone .grad).
    The Taylor half of the hybrid score needs .grad, so when we deepcopy the model to test a candidate
    pruning ratio, we must re-attach the gradients by hand afterwards."""
    m2 = copy.deepcopy(model)
    for p1, p2 in zip(model.parameters(), m2.parameters()):
        p2.grad = p1.grad.clone() if p1.grad is not None else None
    return m2


def make_pruner(model, ex, ratio, ignored):
    return tp.pruner.BasePruner(model, ex, importance=TaylorLAMPImportance(), pruning_ratio=ratio,
                                iterative_steps=1, global_pruning=True, max_pruning_ratio=0.995,
                                ignored_layers=ignored, round_to=None)


def search_ratio(model, ex, target, ignored, iters=14):
    """Bisection: smallest single-shot global ratio whose result has <= `target` params. Uses
    deepcopy_with_grad so every trial scores channels with the SAME accumulated gradients as the real run."""
    lo, hi = 0.0, 0.99
    for _ in range(iters):
        mid = (lo + hi) / 2
        m = deepcopy_with_grad(model).cpu()
        make_pruner(m, ex.cpu(), mid, [m.classifier]).step()
        lo, hi = (lo, mid) if n_params(m) <= target else (mid, hi)
    return hi


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
    v, t = evaluate(student, va, dev), evaluate(student, te, dev)
    rows.append((name, n_params(student), v, t))
    print(f'  -> {name}: params={n_params(student):,}  val_acc={v:.4f}  test_acc={t:.4f}')
    return v, t


@torch.no_grad()
def evaluate_tta(model, loader, dev, use_tta=True, return_preds=False):
    """baseline_lcenet.evaluate() has no TTA support, so this wraps it: with --tta (the default),
    averages the softmax probabilities of each image and its horizontal flip before taking argmax."""
    if not use_tta:
        return evaluate(model, loader, dev, return_preds=return_preds)
    model.eval(); yt, yp = [], []
    for x, y in loader:
        x = x.to(dev)
        probs = model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)
        yp += probs.argmax(1).cpu().tolist(); yt += y.tolist()
    acc = float(np.mean(np.array(yt) == np.array(yp))) if yt else float('nan')
    return (acc, yt, yp) if return_preds else acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./dataset/Potato'); ap.add_argument('--ckpt', default='best_model_cbam.pth')
    ap.add_argument('--ckpt_finetune_layers', default='6,7', help='must match --finetune_layers used to TRAIN --ckpt')
    ap.add_argument('--split', choices=['group', 'repo'], default='group')
    ap.add_argument('--res', type=int, default=224); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--budget', type=int, default=49000, help='final param budget (must be < 50,000)')
    ap.add_argument('--taylor_batches', type=int, default=16, help='real batches used to build the one-shot hybrid score')
    ap.add_argument('--ft_epochs', type=int, default=40, help='single recovery fine-tune after the one-shot cut')
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--out', default='lcenet_pruned_b')
    ap.add_argument('--tta', action='store_true', default=True); ap.add_argument('--no_tta', dest='tta', action='store_false')
    ap.add_argument('--limit', type=int, default=None, help='debug: use only N images')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(0)
    tr, va, te, classes = make_loaders(a.data, a.split, a.res, a.bs, limit=a.limit)
    ex = torch.randn(1, 3, a.res, a.res, device=dev)
    rows = []

    # ---- teacher (original, frozen, unpruned) - identical loading to strategy_a.py
    layers = tuple(x.strip() for x in a.ckpt_finetune_layers.split(','))
    teacher = EfficientNet_CBAM(len(classes), pretrained=not os.path.exists(a.ckpt)).to(dev)
    if os.path.exists(a.ckpt):
        teacher.load_state_dict(torch.load(a.ckpt, map_location=dev))
    else:
        print(f'WARNING: checkpoint "{a.ckpt}" not found - teacher is ImageNet-pretrained only '
              f'(not the fine-tuned LCE-Net), so KD targets will be low quality.')
    for name, param in teacher.named_parameters():
        if not any(layer_name in name for layer_name in layers):
            param.requires_grad = False
    teacher.eval()
    print('[baseline]')
    report('0. Baseline LCE-Net (teacher)', teacher, va, te, dev, rows)

    student = FullNet(teacher).to(dev)
    for p in student.parameters():
        p.requires_grad_(True)

    # ================= Step 1+2: build the hybrid score, on REAL data, right from the start =================
    print(f'\n=== Building Taylor-LAMP hybrid score over {a.taylor_batches} real batches ===')
    accumulate_gradients(student, teacher, tr, dev, a.taylor_batches)

    # ================= Step 3: ONE global cut straight to budget (no separate surgery step) =================
    print('=== Single global cut to budget (Taylor-LAMP hybrid score, one pass) ===')
    ratio = search_ratio(student, ex, a.budget, [student.classifier])
    print(f'  global one-shot ratio {ratio:.4f}')
    # re-accumulate gradients on the actual student (the bisection search only ran on throwaway copies)
    accumulate_gradients(student, teacher, tr, dev, a.taylor_batches)
    make_pruner(student, ex, ratio, [student.classifier]).step()
    print(f'  {n_params(teacher):,} -> {n_params(student):,} params, in a single pruning call')
    report('1. One-shot Taylor-LAMP cut', student, va, te, dev, rows)

    # ================= Recovery fine-tune (not a "phase" of the pruning decision - just training after it) =========
    print('\n=== Recovery fine-tune (knowledge distillation from the original LCE-Net) ===')
    hist = finetune(student, teacher, tr, va, dev, a.ft_epochs, a.lr, 'recovery')
    report('2. Fine-tuned (FINAL)', student, va, te, dev, rows)

    from sklearn.metrics import classification_report
    final_test, yt, yp = evaluate_tta(student, te, dev, use_tta=a.tta, return_preds=True)
    print(f'\nFinal model - TEST SET classification report (held out the entire run{", TTA" if a.tta else ""}):')
    print(classification_report(yt, yp, labels=list(range(len(classes))), target_names=classes, digits=4, zero_division=0))

    print(f'\n{"phase":<30}{"params":>12}{"val acc":>10}{"test acc":>10}')
    for name, p, v, t in rows:
        print(f'{name:<30}{p:>12,}{v:>10.4f}{t:>10.4f}')
    if hist:
        print(f'\nlast-5-epoch mean VAL acc: {np.mean(hist[-5:]):.4f}')
    print(f'FINAL TEST accuracy (report this one{", with TTA" if a.tta else ""}): {final_test:.4f}')

    assert n_params(student) < 50_000, f'{n_params(student):,} params - budget missed; lower --budget'
    student = student.cpu().eval()
    torch.jit.trace(student, torch.randn(1, 3, a.res, a.res)).save(a.out + '.pt')
    print(f'\nOK: {n_params(student):,} params (< 50,000), single-shot cut. Saved {a.out}.pt '
          f'(torch.jit.load; input normalised mean=std=0.5)')

if __name__ == '__main__':
    main()