# Study design: do crystals need higher-order interactions?

## Motivation

Many crystal GNNs add explicit higher-order structure (line graphs, triplets,
hyperedges) on the assumption that ordinary graphs only capture pairwise
interactions. Peixoto, Peel, Gross & De Domenico (arXiv:2602.16937) argue that a
graph fixes which atoms may interact, while the interaction function can be
arbitrarily multivariate. On this view hypergraph models are a restricted
subclass of graph models, not an extension. Whether explicit higher-order
structure is needed is therefore an empirical, task-dependent question.

"Higher-order" can enter a model in two ways, and the study separates them:

- **Structural route:** richer structure (line graph / triplets / hyperedges), simple message function.
  Example: ALIGNN.
- **Functional route:** plain graph, richer multivariate interaction function.
  Example: MACE (body order via tensor products of neighbor sums).

## Research questions

1. **Necessity:** for a given task, does going beyond pairwise improve accuracy?
2. **Route:** if so, is the gain better obtained through structure or through function?
3. **Cost:** which route gives the better accuracy-compute trade-off?

Precise form: *does explicit higher-order structure add value beyond the same
geometric information supplied on a plain graph with a multivariate interaction
function, at matched compute?*

## Phase 1: synthetic potentials with known body order

| potential | true interaction | expected result |
|---|---|---|
| Lennard-Jones (Ar) | pairwise, additive | body order 2 suffices |
| Gupta / EAM (Cu) | many-body via F(sum_j rho(r_ij)), no angles | plain graph + nonlinear function suffices |
| Stillinger-Weber (Si) | explicit 3-body angular term | needs body order >= 3, by either route |

EAM is the key case: genuinely many-body, but needs no hyperedges or triplets.

## Phase 2: real crystal tasks

- Matbench property prediction: formation energy, band gap (likely low-order);
  phonons, shear/bulk moduli (angle-sensitive).
- Force fields: covalent (Si, C) vs metallic (Cu, Al) subsets of MPtrj.

Hypothesis: the need for higher body order is material-class dependent:
high for directional (covalent) bonding, low for metallic/ionic.

## Model grid

| knob | implementation | isolates |
|---|---|---|
| functional body order | `ace --nu 1/2`, `mace --nu 1/2/3` | functional route |
| structural route | `linegraph` (ALIGNN-style, same pipeline) | structural route |
| group structure (planned) | coordination-polyhedra hypergraph vs bipartite incidence graph | hyperedges vs lossless graph encoding |

## Confounds to control

1. **Depth creates effective body order.** Stacked pairwise layers with nonlinear
   updates exceed 2-body. Compare at fixed depth; include single-layer references.
2. **Angular resolution vs body order.** Vary one at a time (MACE: max_L vs nu).
3. **Cutoff radius** identical across models (and equal to the potential's cutoff in Phase 1).
4. **Compute.** Report parameters, wall-clock time, inference cost; matched budgets.
5. **Extra information.** Higher-order models often also receive new features
   (angles, coordination descriptors); compare against a plain graph given the same information.

## Evaluation

- Test error (energy per atom, forces, properties), mean ± std over seeds.
- Learning curves (`--n_train 25 50 100 200 400`): structural priors should matter most with little data.
- Prequential code length (online MDL) for model selection.
- Many-body probe: evaluate trained models on isolated 2/3/4-atom clusters.
  Do **not** read body order from MACE's internal nu-blocks: without purification
  their split into orders is arbitrary (see literature_evidence.md).

## Pilot to confirm the gap

Two Matbench tasks where the literature disagrees (phonons; perovskite formation
energy), 5 seeds, identical cutoff/depth/training:

1. distances only;
2. distances + per-atom angular descriptors (no line graph);
3. line-graph model;
4. equivariant plain graph (nu = 1 vs 2).

If 2 or 4 matches 3, the reported higher-order advantage is mostly information, not structure.
