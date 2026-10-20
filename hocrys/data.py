"""Datasets: synthetic potentials (known body order) and Matbench tasks.

Synthetic: perturbed crystals (random strain + rattle) labelled by the
potentials in potentials.py. Matbench: read the official json.gz files
directly (no `matbench`/`pymatgen` dependency, which keeps ARM/ROCm installs
simple); download them with scripts/get_matbench.sh.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

from . import potentials
from .graphs import build_graph, collate

DATA = Path(__file__).resolve().parents[1] / "data"

SYNTH = {
    # name: (element, lattice, a, supercell, rattle A, potential)
    "lj":  ("Ar", "fcc", 5.26, (2, 2, 2), 0.10, "lj"),
    "eam": ("Cu", "fcc", 3.615, (2, 2, 2), 0.08, "eam"),
    "sw":  ("Si", "diamond", 5.431, (2, 2, 1), 0.10, "sw"),
}

MATBENCH = {
    # name: file stem, cutoff (A). Small tasks first.
    "phonons":    ("matbench_phonons", 5.0),
    "perovskites": ("matbench_perovskites", 5.0),
    "log_gvrh":   ("matbench_log_gvrh", 5.0),
}


def _random_structure(name, rng):
    el, lat, a, rep, rattle, _ = SYNTH[name]
    at = bulk(el, lat, a=a, cubic=True).repeat(rep)
    strain = np.eye(3) + rng.uniform(-0.03, 0.03, (3, 3))
    strain = 0.5 * (strain + strain.T)
    at.set_cell(at.cell @ strain, scale_atoms=True)
    at.positions += rng.normal(0, rattle, at.positions.shape)
    at.wrap()
    return at


def make_synthetic(name, n=600, seed=0):
    """Generate and cache n labelled structures."""
    path = DATA / f"synth_{name}_n{n}_s{seed}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    rng = np.random.default_rng(seed)
    kind = SYNTH[name][-1]
    rc = potentials.CUTOFF[kind]
    graphs = []
    for _ in range(n):
        at = _random_structure(name, rng)
        g = build_graph(at, rc, y=0.0)
        b = collate([g], with_triplets=(kind == "sw"))
        e_pa, f = potentials.label(b, kind)
        g.y = e_pa[0].to(torch.float32)
        g.forces = f.to(torch.float32)
        g.trip = None
        graphs.append(g)
    DATA.mkdir(exist_ok=True)
    torch.save(graphs, path)
    return graphs


def _structure_to_atoms(d):
    """pymatgen Structure dict -> ase Atoms (ordered sites assumed)."""
    cell = np.array(d["lattice"]["matrix"], dtype=float)
    symbols, pos = [], []
    for s in d["sites"]:
        sp = max(s["species"], key=lambda x: x.get("occu", 1.0))
        symbols.append(sp["element"])
        pos.append(s["xyz"])
    return Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)


def load_matbench(name, max_n=None):
    stem, rc = MATBENCH[name]
    path = DATA / "matbench" / f"{stem}.json.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/get_matbench.sh")
    cache = DATA / f"{stem}_rc{rc}.pt"
    if cache.exists():
        graphs = torch.load(cache, weights_only=False)
    else:
        with gzip.open(path, "rt") as fh:
            raw = json.load(fh)
        graphs = [build_graph(_structure_to_atoms(s), rc, y=v) for s, v in raw["data"]]
        torch.save(graphs, cache)
    return graphs[:max_n] if max_n else graphs


def load(name, seed=0, n_synth=600):
    """Return (graphs, cutoff, has_forces)."""
    if name in SYNTH:
        return make_synthetic(name, n_synth, seed=0), potentials.CUTOFF[SYNTH[name][-1]], True
    if name in MATBENCH:
        return load_matbench(name), MATBENCH[name][1], False
    raise ValueError(name)


def split(n, seed, n_train=None, frac=(0.7, 0.15, 0.15)):
    """Fixed val/test per seed; n_train subsamples the train pool (learning curves)."""
    perm = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    if n_train:
        tr = tr[:n_train]
    return tr, va, te
