"""Periodic neighbor lists, triplets and batching shared by every model.

Convention: edge e = (i, j, S) means neighbor j of center i, seen through the
periodic image shifted by integer cell vector S. Edge vector:
    r_ij = pos[j] - pos[i] + S @ cell
All distances/angles are recomputed from `pos` inside the models, so forces
follow from autograd identically for all models.
"""
from dataclasses import dataclass

import numpy as np
import torch
from matscipy.neighbours import neighbour_list


@dataclass
class Graph:
    z: torch.Tensor          # (N,) atomic numbers
    pos: torch.Tensor        # (N,3) Cartesian positions, Angstrom
    cell: torch.Tensor       # (3,3) lattice vectors as rows
    edge_index: torch.Tensor # (2,E) [center i, neighbor j]
    shift: torch.Tensor      # (E,3) integer image shifts
    y: torch.Tensor          # () per-atom energy (eV/atom) or property
    forces: torch.Tensor | None = None  # (N,3) eV/A, None in property mode
    trip: torch.Tensor | None = None    # (2,T) cached line-graph edges


def build_graph(atoms, cutoff, y, forces=None):
    i, j, S = neighbour_list("ijS", atoms, float(cutoff))
    return Graph(
        z=torch.as_tensor(atoms.numbers, dtype=torch.long),
        pos=torch.as_tensor(atoms.positions, dtype=torch.float32),
        cell=torch.as_tensor(np.asarray(atoms.cell), dtype=torch.float32),
        edge_index=torch.as_tensor(np.stack([i, j]), dtype=torch.long),
        shift=torch.as_tensor(S, dtype=torch.float32),
        y=torch.as_tensor(float(y), dtype=torch.float32),
        forces=None if forces is None else torch.as_tensor(forces, dtype=torch.float32),
    )


def triplets(edge_index, num_nodes):
    """Ordered pairs of distinct edges sharing the same center atom.

    Returns (2,T) tensor [e1, e2]: e1=(i,j), e2=(i,k), e1 != e2.
    This is the line graph used by the structural (ALIGNN-style) route; its
    size grows as sum_i deg(i)*(deg(i)-1), which is the cost being measured.
    """
    center = edge_index[0]
    order = torch.argsort(center, stable=True)
    deg = torch.bincount(center, minlength=num_nodes)
    starts = torch.cumsum(deg, 0) - deg
    e1s, e2s = [], []
    for d in torch.unique(deg):
        d = int(d)
        if d < 2:
            continue
        nodes = torch.nonzero(deg == d).flatten()
        base = starts[nodes]                                  # (n,)
        a, b = torch.meshgrid(torch.arange(d), torch.arange(d), indexing="ij")
        mask = a != b
        a, b = a[mask], b[mask]                                # (d*(d-1),)
        e1s.append(order[(base[:, None] + a[None]).flatten()])
        e2s.append(order[(base[:, None] + b[None]).flatten()])
    if not e1s:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.stack([torch.cat(e1s), torch.cat(e2s)])


@dataclass
class Batch:
    z: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_offset: torch.Tensor  # (E,3) Cartesian image offset S @ cell
    shift: torch.Tensor        # (E,3) integer shifts (for MACE)
    cell: torch.Tensor         # (G*3,3) stacked cells (for MACE)
    batch: torch.Tensor        # (N,) graph id per atom
    ptr: torch.Tensor          # (G+1,)
    y: torch.Tensor            # (G,)
    forces: torch.Tensor | None
    num_graphs: int
    triplets: torch.Tensor | None = None

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self

    def edge_vectors(self):
        i, j = self.edge_index
        return self.pos[j] - self.pos[i] + self.edge_offset


def collate(graphs, with_triplets=False):
    zs, pos, ei, off, sh, cells, bvec, ys, fs, tr = [], [], [], [], [], [], [], [], [], []
    ptr, eptr = [0], 0
    for g_id, g in enumerate(graphs):
        n = g.z.numel()
        zs.append(g.z); pos.append(g.pos); cells.append(g.cell)
        ei.append(g.edge_index + ptr[-1])
        off.append(g.shift @ g.cell); sh.append(g.shift)
        bvec.append(torch.full((n,), g_id, dtype=torch.long))
        ys.append(g.y)
        if g.forces is not None:
            fs.append(g.forces)
        if with_triplets:
            if g.trip is None:  # computed once per structure, then cached
                g.trip = triplets(g.edge_index, n)
            tr.append(g.trip + eptr)
        eptr += g.edge_index.shape[1]
        ptr.append(ptr[-1] + n)
    edge_index = torch.cat(ei, 1)
    b = Batch(
        z=torch.cat(zs), pos=torch.cat(pos), edge_index=edge_index,
        edge_offset=torch.cat(off), shift=torch.cat(sh), cell=torch.cat(cells),
        batch=torch.cat(bvec), ptr=torch.tensor(ptr), y=torch.stack(ys),
        forces=torch.cat(fs) if len(fs) == len(graphs) else None,
        num_graphs=len(graphs),
    )
    if with_triplets:
        b.triplets = torch.cat(tr, 1)
    return b
