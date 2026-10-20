"""Four models that differ ONLY in how many-body interactions enter.

All share: element embedding, Bessel radial basis with the same cutoff,
width F, number of layers T, per-atom readout. Each returns per-atom
contributions; energy/property = mean over atoms, forces = -dE/dpos.

  pair      : distances only, 2-body messages (SchNet/CGCNN-like)      [baseline]
  ace       : PLAIN graph, angular info via a multivariate node function:
              A_ilm = sum_j R(r_ij) h_j Y_lm(r_ij);  nu=1 -> A_00 only (2-body),
              nu=2 -> + |A_l|^2 power spectrum (3-body)               [functional route]
  linegraph : ALIGNN-style, explicit triplets with angle features       [structural route]
  mace      : official MACE, correlation order nu (body order nu+1)     [reference SOTA]
"""
import math

import torch
from torch import nn

from .graphs import Batch


def scatter_sum(src, index, n):
    out = torch.zeros((n,) + src.shape[1:], dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


class Radial(nn.Module):
    """Bessel basis x polynomial envelope (as in DimeNet/MACE)."""

    def __init__(self, cutoff, n=8, p=6):
        super().__init__()
        self.rc, self.p = cutoff, p
        self.register_buffer("freq", torch.arange(1, n + 1) * math.pi / cutoff)

    def envelope(self, r):
        """Smooth polynomial envelope: 1 at r=0, 0 (with derivatives) at cutoff."""
        x = (r / self.rc).clamp(max=1.0)[:, None]
        p = self.p
        return 1 - (p + 1) * (p + 2) / 2 * x ** p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2)

    def forward(self, r):
        return math.sqrt(2 / self.rc) * torch.sin(self.freq * r[:, None]) / r[:, None] * self.envelope(r)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


def real_sh(u):
    """Real spherical harmonics l=1,2 (unnormalised), u unit vectors (E,3)."""
    x, y, z = u.unbind(-1)
    l1 = torch.stack([x, y, z], -1)
    l2 = torch.stack([x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y], -1) * torch.tensor(
        [math.sqrt(3), math.sqrt(3), 0.5, math.sqrt(3), 0.5 * math.sqrt(3)], device=u.device)
    return [l1, l2]


class Base(nn.Module):
    needs_triplets = False

    def __init__(self, cutoff, F=64, T=3, n_rbf=8):
        super().__init__()
        self.emb = nn.Embedding(100, F)
        self.rbf = Radial(cutoff, n_rbf)
        self.out = mlp(F, F, 1)
        self.F, self.T = F, T

    def atom_out(self, b: Batch):  # (N,) per-atom contributions
        raise NotImplementedError

    def forward(self, b: Batch):
        return self.atom_out(b)


class Pair(Base):
    def __init__(self, cutoff, F=64, T=3, n_rbf=8, **_):
        super().__init__(cutoff, F, T, n_rbf)
        self.filt = nn.ModuleList(mlp(n_rbf, F, F) for _ in range(T))
        self.upd = nn.ModuleList(mlp(F, F, F) for _ in range(T))

    def atom_out(self, b):
        i, j = b.edge_index
        r = b.edge_vectors().norm(dim=-1)
        e, u = self.rbf(r), self.rbf.envelope(r)
        h = self.emb(b.z)
        for t in range(self.T):
            m = scatter_sum(h[j] * self.filt[t](e) * u, i, h.shape[0])
            h = h + self.upd[t](m)
        return self.out(h).squeeze(-1)


class ACE(Base):
    """Plain graph + multivariate node function (functional route)."""

    def __init__(self, cutoff, F=64, T=3, n_rbf=8, nu=2, **_):
        super().__init__(cutoff, F, T, n_rbf)
        self.nu = nu
        n_inv = 1 if nu == 1 else 3  # A_0 ; + |A_1|^2, |A_2|^2
        self.filt = nn.ModuleList(mlp(n_rbf, F, F) for _ in range(T))
        self.upd = nn.ModuleList(mlp(n_inv * F, F, F) for _ in range(T))

    def atom_out(self, b):
        i, j = b.edge_index
        vec = b.edge_vectors()
        r = vec.norm(dim=-1)
        e, u = self.rbf(r), self.rbf.envelope(r)
        Y = real_sh(vec / r[:, None]) if self.nu >= 2 else []
        h = self.emb(b.z)
        n = h.shape[0]
        for t in range(self.T):
            w = h[j] * self.filt[t](e) * u                   # (E,F)
            feats = [scatter_sum(w, i, n)]                   # A_0: 2-body
            for Yl in Y:                                      # |A_l|^2: 3-body
                A = scatter_sum(w[:, :, None] * Yl[:, None, :], i, n)  # (N,F,2l+1)
                feats.append((A ** 2).sum(-1) / (h.shape[1] ** 0.5))
            h = h + self.upd[t](torch.cat(feats, -1))
        return self.out(h).squeeze(-1)


class LineGraph(Base):
    """ALIGNN-style: edge states updated on the line graph with bond angles."""
    needs_triplets = True

    def __init__(self, cutoff, F=64, T=3, n_rbf=8, n_ang=8, **_):
        super().__init__(cutoff, F, T, n_rbf)
        self.edge_emb = mlp(n_rbf, F, F)
        self.n_ang = n_ang
        self.ang = nn.ModuleList(mlp(n_ang + 2 * F, F, F) for _ in range(T))
        self.gate = nn.ModuleList(nn.Linear(n_ang + 2 * F, F) for _ in range(T))
        self.filt = nn.ModuleList(mlp(F, F, F) for _ in range(T))
        self.upd = nn.ModuleList(mlp(F, F, F) for _ in range(T))

    def angle_basis(self, cos):
        c = torch.linspace(-1, 1, self.n_ang, device=cos.device)
        return torch.exp(-((cos[:, None] - c) ** 2) / (2 * 0.15 ** 2))

    def atom_out(self, b):
        i, j = b.edge_index
        vec = b.edge_vectors()
        r = vec.norm(dim=-1)
        rb, u = self.rbf(r), self.rbf.envelope(r)
        e1, e2 = b.triplets
        cos = (vec[e1] * vec[e2]).sum(-1) / (r[e1] * r[e2])
        a = self.angle_basis(cos.clamp(-1, 1))
        h = self.emb(b.z)
        x = self.edge_emb(rb)                                     # edge states
        for t in range(self.T):
            z = torch.cat([a, x[e1], x[e2]], -1)
            msg = torch.sigmoid(self.gate[t](z)) * self.ang[t](z) * u[e2]  # triplet message
            x = x + scatter_sum(msg, e1, x.shape[0])
            m = scatter_sum(h[j] * self.filt[t](x) * u, i, h.shape[0])
            h = h + self.upd[t](m)
        return self.out(h).squeeze(-1)


class MACEWrap(nn.Module):
    """Official MACE (mace-torch); only correlation order nu is varied."""

    def __init__(self, cutoff, F=64, T=2, nu=2, elements=(14,), avg_neighbors=10.0, max_ell=2, **_):
        super().__init__()
        torch.serialization.add_safe_globals([slice])  # e3nn constants file on torch>=2.6
        from e3nn import o3
        from mace.modules import MACE, RealAgnosticInteractionBlock, RealAgnosticResidualInteractionBlock

        self.elements = list(elements)
        self.register_buffer("zmap", torch.full((120,), -1, dtype=torch.long))
        self.zmap[torch.tensor(self.elements)] = torch.arange(len(self.elements))
        self.model = MACE(
            r_max=cutoff, num_bessel=8, num_polynomial_cutoff=6, max_ell=max_ell,
            interaction_cls=RealAgnosticResidualInteractionBlock,
            interaction_cls_first=RealAgnosticInteractionBlock,
            num_interactions=T, num_elements=len(self.elements),
            hidden_irreps=o3.Irreps(f"{F}x0e"), MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=torch.zeros(len(self.elements)), avg_num_neighbors=avg_neighbors,
            atomic_numbers=self.elements, correlation=nu, gate=torch.nn.functional.silu,
        )

    needs_triplets = False

    def forward(self, b: Batch):
        idx = self.zmap[b.z]
        node_attrs = torch.nn.functional.one_hot(idx, len(self.elements)).to(b.pos.dtype)
        i, j = b.edge_index
        data = {
            "positions": b.pos, "node_attrs": node_attrs,
            # MACE: vectors = pos[receiver] - pos[sender] + shifts; sender=j, receiver=i
            "edge_index": torch.stack([j, i]), "shifts": -b.edge_offset, "unit_shifts": -b.shift,
            "cell": b.cell, "batch": b.batch, "ptr": b.ptr,
            "head": torch.zeros(b.num_graphs, dtype=torch.long, device=b.pos.device),
        }
        out = self.model(data, training=self.training, compute_force=False)
        return out["node_energy"]


MODELS = {"pair": Pair, "ace": ACE, "linegraph": LineGraph, "mace": MACEWrap}


def build(name, cutoff, **kw):
    return MODELS[name](cutoff, **kw)
