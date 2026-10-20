"""Classical potentials with KNOWN interaction order (ground truth for Phase 1).

    lj  : Lennard-Jones (Ar-like)        -> pairwise, additive (body order 2)
    eam : Gupta / second-moment TB (Cu)  -> many-body, NO angles: E_i = -sqrt(sum_j f(r_ij))
    sw  : Stillinger-Weber (Si)          -> explicit 3-body angular term

Implemented in torch on the shared graph, so energies and forces (autograd)
are exact for the same neighbor list the models see.
"""
import math

import torch

from .graphs import Batch


def _switch(r, r_on, r_off):
    """Smooth cosine switch: 1 below r_on, 0 above r_off."""
    x = ((r - r_on) / (r_off - r_on)).clamp(0, 1)
    return 0.5 * (1 + torch.cos(math.pi * x))


def _scatter(src, index, n):
    return torch.zeros(n, dtype=src.dtype).index_add_(0, index, src)


# Cutoffs used both for the potential and (identically) for every model.
CUTOFF = {"lj": 5.6, "eam": 4.0, "sw": 3.77}


def lj_energy(b: Batch):
    eps, sig = 0.0104, 3.40
    r = b.edge_vectors().norm(dim=-1)
    sr6 = (sig / r) ** 6
    e = 4 * eps * (sr6 ** 2 - sr6) * _switch(r, 4.8, CUTOFF["lj"])
    return 0.5 * _scatter(e, b.edge_index[0], b.pos.shape[0])  # per atom


def eam_energy(b: Batch):
    A, xi, p, q, r0 = 0.0855, 1.224, 10.960, 2.278, 2.556  # Cleri-Rosato Cu
    r = b.edge_vectors().norm(dim=-1)
    sw = _switch(r, 3.6, CUTOFF["eam"])
    n = b.pos.shape[0]
    i = b.edge_index[0]
    rep = _scatter(A * torch.exp(-p * (r / r0 - 1)) * sw, i, n)
    rho = _scatter(xi ** 2 * torch.exp(-2 * q * (r / r0 - 1)) * sw, i, n)
    return rep - torch.sqrt(rho + 1e-12)


def sw_energy(b: Batch, trip):
    A, B, p, a, lam, gam, sig, eps = 7.049556277, 0.6022245584, 4.0, 1.80, 21.0, 1.20, 2.0951, 2.1683
    rc = a * sig
    vec = b.edge_vectors()
    r = vec.norm(dim=-1)
    n = b.pos.shape[0]
    i = b.edge_index[0]
    inside = r < rc - 1e-6
    rs = torch.where(inside, r / sig, torch.full_like(r, 1.0))
    expo = torch.where(inside, torch.exp(1.0 / (rs - a)), torch.zeros_like(r))
    v2 = A * eps * (B * rs ** (-p) - 1) * expo
    e = 0.5 * _scatter(v2, i, n)
    # three-body: sum over ordered pairs counts each (j,k) twice -> factor 1/2
    e1, e2 = trip
    cos = (vec[e1] * vec[e2]).sum(-1) / (r[e1] * r[e2])
    g = torch.where(inside, torch.exp(gam / (rs - a)), torch.zeros_like(r))
    v3 = lam * eps * (cos + 1.0 / 3.0) ** 2 * g[e1] * g[e2]
    return e + 0.5 * _scatter(v3, i[e1], n)


def label(b: Batch, kind):
    """Return per-graph energy per atom and per-atom forces."""
    b.pos.requires_grad_(True)
    if kind == "lj":
        ea = lj_energy(b)
    elif kind == "eam":
        ea = eam_energy(b)
    elif kind == "sw":
        ea = sw_energy(b, b.triplets)
    else:
        raise ValueError(kind)
    (grad,) = torch.autograd.grad(ea.sum(), b.pos)
    counts = torch.bincount(b.batch, minlength=b.num_graphs)
    e_pa = _scatter(ea.detach(), b.batch, b.num_graphs) / counts
    b.pos.requires_grad_(False)
    return e_pa, -grad
