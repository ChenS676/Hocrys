"""Physics sanity checks. Run: python -m pytest tests -q   (or python tests/test_sanity.py)"""
import sys
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hocrys import potentials  # noqa: E402
from hocrys.graphs import build_graph, collate  # noqa: E402
from hocrys.models import build  # noqa: E402

torch.set_default_dtype(torch.float32)
MODELS = [("pair", {}), ("ace", {"nu": 2}), ("linegraph", {}), ("mace", {"nu": 2, "T": 2, "elements": [14]})]


def si_cell(rattle=0.05, seed=0, rep=(1, 1, 1)):
    at = bulk("Si", "diamond", a=5.431, cubic=True).repeat(rep)
    at.positions += np.random.default_rng(seed).normal(0, rattle, at.positions.shape)
    return at


def energy(model, at, rc, need_trip):
    b = collate([build_graph(at, rc, 0.0)], with_triplets=need_trip)
    b.pos = b.pos.double()
    b.edge_offset = b.edge_offset.double()
    b.cell = b.cell.double()
    return model(b).sum()


def test_sw_reference_energy():
    """Perfect diamond Si with Stillinger-Weber: -4.3366 eV/atom (2*eps)."""
    at = si_cell(rattle=0.0)
    b = collate([build_graph(at, potentials.CUTOFF["sw"], 0.0)], with_triplets=True)
    e, f = potentials.label(b, "sw")
    assert abs(e.item() + 4.3366) < 2e-3, e.item()
    assert f.abs().max() < 1e-4


def test_models_invariance_and_forces():
    rc = potentials.CUTOFF["sw"]
    at = si_cell()
    R = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(1)))[0].numpy()
    for name, kw in MODELS:
        torch.manual_seed(0)
        m = build(name, rc, **kw).double().eval()
        nt = getattr(m, "needs_triplets", False)
        e0 = energy(m, at, rc, nt)
        # rotation + translation invariance
        at_r = at.copy()
        at_r.set_cell(at.cell @ R.T, scale_atoms=False)
        at_r.positions = at.positions @ R.T + 0.37
        assert torch.allclose(e0, energy(m, at_r, rc, nt), atol=1e-6), name
        # supercell: energy per atom unchanged
        e2 = energy(m, at.repeat((2, 1, 1)), rc, nt)
        assert torch.allclose(e0 / len(at), e2 / (2 * len(at)), atol=1e-6), name
        # forces = -grad matches finite differences
        b = collate([build_graph(at, rc, 0.0)], with_triplets=nt)
        b.pos, b.edge_offset, b.cell = b.pos.double().requires_grad_(), b.edge_offset.double(), b.cell.double()
        (g,) = torch.autograd.grad(m(b).sum(), b.pos)
        h = 1e-5
        for a_idx, d in [(0, 0), (3, 2)]:
            ap, am = at.copy(), at.copy()
            ap.positions[a_idx, d] += h
            am.positions[a_idx, d] -= h
            fd = (energy(m, ap, rc, nt) - energy(m, am, rc, nt)) / (2 * h)
            assert abs(fd.item() - g[a_idx, d].item()) < 1e-4 * max(1, abs(fd.item())), (name, fd.item(), g[a_idx, d].item())
        print(f"  ok  {name}")


def test_angle_sensitivity_matches_design():
    """Body-order check on a single center atom (T=1 layer):
    rotating neighbor c around atom a at fixed |ac| changes the bond angle.
    Atom a's output must ignore the angle for pair and ace-nu1 (2-body),
    and must depend on it for ace-nu2 and linegraph (3-body)."""
    from ase import Atoms

    rc = 3.0

    def atom0(m, nt, th):
        pos = [[0, 0, 0], [2.3, 0, 0], [2.3 * np.cos(th), 2.3 * np.sin(th), 0]]
        at = Atoms("Si3", positions=pos, cell=[40, 40, 40], pbc=True)
        b = collate([build_graph(at, rc, 0.0)], with_triplets=nt)
        b.pos, b.edge_offset, b.cell = b.pos.double(), b.edge_offset.double(), b.cell.double()
        return m(b)[0].item()

    expect = {"pair": False, "ace1": False, "ace2": True, "linegraph": True}
    for key, (name, kw) in {"pair": ("pair", {}), "ace1": ("ace", {"nu": 1}),
                            "ace2": ("ace", {"nu": 2}), "linegraph": ("linegraph", {})}.items():
        torch.manual_seed(0)
        m = build(name, rc, T=1, **kw).double().eval()
        nt = getattr(m, "needs_triplets", False)
        # th=1.6 vs 2.2 rad: |bc| stays < rc? b-c distance ~3.3/4.1 > rc, so b,c are not neighbors
        diff = abs(atom0(m, nt, 1.6) - atom0(m, nt, 2.2))
        assert (diff > 1e-9) == expect[key], (key, diff)
        print(f"  ok  angle sensitivity {key}: {diff:.2e}")


if __name__ == "__main__":
    test_sw_reference_energy()
    print("  ok  SW reference energy")
    test_models_invariance_and_forces()
    test_angle_sensitivity_matches_design()
    print("all tests passed")
