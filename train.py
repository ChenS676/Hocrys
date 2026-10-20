"""One training loop for every model x dataset. Appends one row to results/results.csv.

Examples:
  python train.py --data sw --model ace --nu 2 --seed 0
  python train.py --data sw --model mace --nu 3 --n_train 100
  python train.py --data phonons --model linegraph --epochs 200
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from hocrys import data as D
from hocrys.graphs import collate
from hocrys.models import build

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="lj | eam | sw | phonons | perovskites | log_gvrh")
p.add_argument("--model", required=True, help="pair | ace | linegraph | mace")
p.add_argument("--nu", type=int, default=2, help="correlation order (ace, mace)")
p.add_argument("--F", type=int, default=64)
p.add_argument("--T", type=int, default=3, help="layers (mace uses T-1, see README)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n_train", type=int, default=None, help="subsample train pool (learning curves)")
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--w_force", type=float, default=10.0)
p.add_argument("--patience", type=int, default=40)
p.add_argument("--max_n", type=int, default=None, help="cap dataset size (smoke tests)")
p.add_argument("--out", default="results/results.csv")
p.add_argument("--wandb", action="store_true")
p.add_argument("--log_every", type=int, default=10, help="batch W&B logging interval (epochs)")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

graphs, cutoff, has_forces = D.load(args.data)
if args.max_n:
    graphs = graphs[: args.max_n]
tr, va, te = D.split(len(graphs), args.seed, args.n_train)
train_g = [graphs[k] for k in tr]

ys = torch.stack([g.y for g in train_g])
mu, sd = ys.mean().item(), ys.std().item() + 1e-8
f_rms = torch.cat([g.forces for g in train_g]).pow(2).mean().sqrt().item() if has_forces else 1.0
elements = sorted({int(z) for g in graphs for z in g.z})
avg_nb = float(np.mean([g.edge_index.shape[1] / g.z.numel() for g in train_g]))

kw = dict(F=args.F, T=args.T, nu=args.nu)
if args.model == "mace":  # 2 MACE layers ~ budget of 3 simple layers
    kw.update(T=max(args.T - 1, 1), elements=elements, avg_neighbors=avg_nb)
model = build(args.model, cutoff, **kw).to(dev)
need_trip = getattr(model, "needs_triplets", False)
n_params = sum(p.numel() for p in model.parameters())

opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.patience // 4)


def batches(idx, shuffle):
    idx = np.random.permutation(idx) if shuffle else idx
    for k in range(0, len(idx), args.bs):
        yield collate([graphs[m] for m in idx[k : k + args.bs]], with_triplets=need_trip).to(dev)


def predict(b, train):
    """Returns (y_hat per graph, forces or None), in physical units."""
    use_f = has_forces
    if use_f:
        b.pos.requires_grad_(True)
    atom = model(b) * sd + mu                                 # per-atom contribution
    counts = torch.bincount(b.batch, minlength=b.num_graphs)
    y_hat = torch.zeros(b.num_graphs, device=dev).index_add_(0, b.batch, atom) / counts
    f_hat = None
    if use_f:
        (g,) = torch.autograd.grad(atom.sum(), b.pos, create_graph=train)
        f_hat = -g
    return y_hat, f_hat


def run(idx, train):
    model.train(train)
    tot, n_g, ae_y, ae_f, n_f = 0.0, 0, 0.0, 0.0, 0
    for b in batches(idx, shuffle=train):
        y_hat, f_hat = predict(b, train)
        loss = ((y_hat - b.y) / sd).pow(2).mean()
        if f_hat is not None:
            loss = loss + args.w_force * ((f_hat - b.forces) / f_rms).pow(2).mean()
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        tot += loss.item() * b.num_graphs
        n_g += b.num_graphs
        ae_y += (y_hat - b.y).abs().sum().item()
        if f_hat is not None:
            ae_f += (f_hat - b.forces).abs().sum().item()
            n_f += f_hat.numel()
    return tot / n_g, ae_y / n_g, (ae_f / n_f if n_f else float("nan"))


wb = None
if args.wandb:
    import wandb
    wb = wandb.init(project="hocrys-bench", config=vars(args))
buf = []

tag = f"{args.data}/{args.model}" + (f"-nu{args.nu}" if args.model in ("ace", "mace") else "")
print(f"{tag}  params={n_params}  train={len(tr)} val={len(va)} test={len(te)}  dev={dev}", flush=True)
best, best_state, wait, t0 = float("inf"), None, 0, time.time()
for ep in range(1, args.epochs + 1):
    tl, _, _ = run(tr, True)
    vl, vy, vf = run(va, False)
    sched.step(vl)
    if vl < best:
        best, wait = vl, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
    print(f"\r  ep {ep:4d}  train {tl:.4f}  val {vl:.4f}  val_mae_y {vy:.4g}  val_mae_f {vf:.4g}", end="", flush=True)
    buf.append({"epoch": ep, "train_loss": tl, "val_loss": vl, "val_mae_y": vy, "val_mae_f": vf})
    if wb and ep % args.log_every == 0:
        for row in buf:
            wb.log(row)
        buf.clear()
    if wait >= args.patience:
        break
train_time = time.time() - t0
if wb:
    for row in buf:
        wb.log(row)

model.load_state_dict(best_state)
t1 = time.time()
_, te_y, te_f = run(te, False)
infer_time = (time.time() - t1) / max(len(te), 1)
print(f"\n  TEST  mae_y {te_y:.5g}  mae_f {te_f:.5g}  ({train_time:.0f}s, {ep} ep)", flush=True)

row = dict(data=args.data, model=args.model, nu=args.nu if args.model in ("ace", "mace") else "",
           T=kw["T"], F=args.F, seed=args.seed, n_train=len(tr), params=n_params,
           test_mae_y=te_y, test_mae_f=te_f, epochs=ep, train_s=round(train_time, 1),
           infer_ms_per_struct=round(1000 * infer_time, 2), device=dev)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
new = not out.exists()
with out.open("a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
if wb:
    wb.summary.update(row)
    wb.finish()
