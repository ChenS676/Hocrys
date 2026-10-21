# HOCrys-Bench (pilot)

Do crystal ML models need **explicit higher-order structure**, or does a plain
graph with a richer interaction function suffice? This pilot compares the two
routes under one shared pipeline: same neighbor lists, cutoff, radial basis,
width, training loop and metrics. Only the way many-body interactions enter differs.

## Models

| name | route | body order per layer | notes |
|---|---|---|---|
| `pair` | baseline | 2 | SchNet/CGCNN-like, distances only |
| `ace --nu 1` | functional | 2 | plain graph, A_0 = sum_j R(r) h_j |
| `ace --nu 2` | functional | 3 | plain graph + power spectrum \|A_l\|^2 (l=1,2): angles without triplets |
| `linegraph` | structural | 3 | ALIGNN-style: explicit triplets (line graph) with angle features |
| `mace --nu k` | reference | k+1 | official `mace-torch`, only correlation order varied |

All models output per-atom contributions; energy/property = mean over atoms,
forces = -dE/dpos by autograd (identical for all models). MACE uses T-1 layers
(2 by default) because each MACE layer is heavier; report `params`/time with results.

## Datasets

| name | ground truth | what it tests |
|---|---|---|
| `lj` | Lennard-Jones Ar (pairwise) | 2-body should suffice |
| `eam` | Gupta/second-moment Cu: E_i = -sqrt(sum_j f(r)) | many-body **without angles**: plain graph should suffice |
| `sw` | Stillinger-Weber Si (3-body angular) | needs body order >= 3 by either route |
| `phonons`, `perovskites`, `log_gvrh` | Matbench | real properties (phonons/shear expected angle-sensitive) |

Synthetic data (600 strained/rattled 32-atom cells) are generated on first use
and cached in `data/`. Potentials and models share the same cutoff per dataset.
Matbench files: `bash scripts/get_matbench.sh` (official json.gz, parsed
without `matbench`/`pymatgen`). The pilot uses a 70/15/15 split per seed, not the
official 5-fold protocol; switch to that before publishing numbers.

## Run

```bash
pip install -r requirements.txt
python tests/test_sanity.py                 # invariance, forces vs finite diff, supercell, body-order checks
python train.py --data sw --model ace --nu 2 --seed 0
python train.py --data sw --model linegraph --n_train 100   # learning-curve point
bash scripts/run_pilot.sh                    # full pilot grid (90 jobs)
sbatch --array=0-89 scripts/slurm_array.sh   # same grid on SLURM
python scripts/summarize.py                  # mean +- std over seeds
```

Each run appends one row to `results/results.csv`: test MAE (per-atom energy or
property; forces in eV/A), params, epochs, train time, inference ms/structure.
Optional `--wandb` logs in batches every `--log_every` epochs.

## Smoke-test results (CPU, 200 train structures, 30 epochs, 1 seed — not converged)

Force MAE (eV/A); train-set force RMS in brackets.

| model | LJ (0.050) | EAM (0.90) | SW (2.29) |
|---|---|---|---|
| pair | 0.0058 | 0.085 | 0.43 |
| ace-nu1 | | | 0.43 |
| ace-nu2 | 0.0034 | 0.055 | **0.072** |
| linegraph | | | 0.20 (210 s vs 55 s for ace-nu2) |
| mace-nu1 | | | 0.40 |
| mace-nu2 | | | 0.14 |

Already the expected pattern: on SW all 2-body models stall at ~0.4 eV/A, and
3-body angular information fixes it, through the plain-graph route at a fraction
of the line-graph cost. Treat as a plumbing check only; run the grid with
seeds and convergence before drawing conclusions.

## Documentation

- `docs/study_design.md`: research questions, phases, model grid, confounds, evaluation
- `docs/literature_evidence.md`: what existing work shows and why it does not settle the question
- `docs/eurohpc_application.md`: compute proposal text (JUPITER Booster / LUMI-G)

## Layout

```
hocrys/graphs.py      periodic neighbor lists (matscipy), cached triplets, batching
hocrys/potentials.py  LJ / Gupta-EAM / Stillinger-Weber in torch (labels + forces)
hocrys/data.py        synthetic generation, Matbench loader, splits
hocrys/models.py      pair, ace, linegraph, mace wrapper
train.py              single training/eval loop for everything
tests/test_sanity.py  physics checks
scripts/              get_matbench.sh, run_pilot.sh, slurm_array.sh, summarize.py
docs/                 study design, literature evidence, compute proposal
```

## Next steps (not in the pilot)

- Learning curves: loop `--n_train 25 50 100 200 400`.
- Many-body probe: evaluate trained models on isolated 2/3/4-atom clusters
  (never read body order from MACE's internal nu-blocks).
- Prequential code length (MDL) for model selection.
- A hypergraph model over coordination polyhedra vs its bipartite encoding.
