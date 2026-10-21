# EuroHPC Development Access application (text as drafted)

Partition: JUPITER Booster (fallback: LUMI-G). Duration: 6 months.

## Project title

Do crystals need higher-order interactions? A controlled benchmark of structural
and functional many-body modelling in graph neural networks

## Abstract

Graph neural networks for crystals increasingly add explicit higher-order
structure, such as bond-angle line graphs, triplets and hyperedges, assuming that
many-body interactions require it. Recent theory shows that graphs with
multivariate interaction functions are at least as expressive, so whether
explicit higher-order structure is necessary is an empirical question. We test
it with controlled experiments. First, synthetic data from classical potentials
with known body order (Lennard-Jones, EAM, Stillinger-Weber) checks whether
models recover the true interaction order. Second, real crystal tasks cover
property prediction (Matbench) and force fields for covalent and metallic
materials. We vary body order through structure and through function
independently, at matched compute, and compare them using accuracy, learning
curves, MDL-based model selection and many-body decomposition. Code and results
will be released openly.

## Keywords

graph neural networks; higher-order interactions; many-body interactions; crystal
materials; machine learning interatomic potentials; model selection; minimum
description length; benchmarking

## Research fields

Computer science: machine learning / AI 60%; materials science and condensed matter 40%.

## Resources (JUPITER Booster)

| field | value |
|---|---|
| Codes | PyTorch, PyTorch Geometric, MACE, MatGL, ASE, LAMMPS, HOCrys-Bench |
| Node hours | 300 |
| Processes/threads | 4 processes / 72 threads (4 independent single-GPU trainings per node) |
| Job memory | 200 GB |
| Max memory per process | 32,000 MB |
| Data transfer | 50 GB |
| Storage | 300 GB |

| work package | node hours |
|---|---|
| Phase 1: synthetic potentials | 40 |
| Phase 2a: Matbench property prediction | 75 |
| Phase 2b: force fields (Si, C, Cu, Al) | 50 |
| Prequential (MDL) evaluation | 60 |
| Many-body decomposition and evaluation | 25 |
| Debugging and reruns | 50 |

LUMI-G variant: 350 node hours, 8 processes / 56 threads, 24,000 MB per process,
ROCm containers, two weeks of porting/validation on ROCm.

## Code details and development

**Description.** HOCrys-Bench is a PyTorch framework that compares crystal graph
neural networks differing only in how many-body interactions enter: a pairwise
baseline, a plain-graph model with a multivariate (angular power-spectrum) node
function, an ALIGNN-style line-graph model with explicit triplets, and MACE with
variable correlation order. All models share one data pipeline, so differences
reflect only the interaction route. Ground-truth labels come from classical
potentials with known body order implemented in PyTorch, plus Matbench tasks.
Work is parallelized as independent single-GPU trainings distributed by job
arrays. The main bottleneck is triplet enumeration in the line-graph model,
mitigated by caching triplets per structure.

**Version:** HOCrys-Bench 0.1 (pilot). **License:** MIT. **Connection:** main developer.

**Scalability.** Many independent small trainings; each fits on one GPU, so
throughput scales near-linearly with GPUs and needs no inter-process
communication. CPU pilot: 30-epoch training on 200 Si structures takes 17-210 s
depending on the model; the line-graph model is about 4x slower than the
plain-graph angular model at comparable or worse accuracy.

**Target.** Full grid (about 400-600 trainings incl. learning curves, 3 seeds)
within the allocation; each training under 4 h on one GPU; 4 per node.

**Main algorithms.** (1) pairwise continuous-filter convolutions; (2) neighbor
sums projected on spherical harmonics (l = 0-2) combined via their power spectrum,
giving three-body angular information without triplets, linear in edges;
(3) line-graph updates over bond-pair triplets, quadratic in neighbors per atom;
(4) MACE tensor-product many-body features, correlation order 1-3. Forces by
automatic differentiation. Task-level parallelism via SLURM job arrays.

**Bottlenecks.** Triplet enumeration and message passing; second-order autodiff
for force training; equivariant tensor products; many small jobs underutilizing GPUs.

**Solutions.** Cached triplets; precomputed neighbor lists (matscipy); triplet-free
angular model; four jobs per node; tuned batch sizes; bf16 where force accuracy allows.

**Enabling work.** Port/validate the stack on the target system (aarch64 containers
on JUPITER); move batching to GPU; profile line-graph and MACE models; reproduce
CPU reference tests on GPU.

**Limitation addressed.** The CPU pilot is too slow for the full grid of models,
datasets, training-set sizes and seeds needed for statistically sound conclusions.

**Contribution to Tier-0.** Identifies the many-body modelling route with the best
accuracy-compute trade-off, informing architecture choice and sizing for future
large-scale training of universal interatomic potentials (MPtrj, OMat24).

**Impact.** Built on widely used open-source codes; the controlled-comparison
design transfers to molecules, surfaces and other graph domains where the
graph-vs-hypergraph question arises; modular, extensible via pull requests.

**Application Support Team:** not needed (JUPITER); limited support for ROCm porting of e3nn (LUMI-G).

## Ethics (100 words)

The project uses only public materials databases and synthetic data from
classical potentials. It involves no personal data or human participants. The
work is purely civilian basic research on materials modelling. It is designed to
test whether added model complexity is justified by evidence, and negative
results will be reported openly. Models are compared at matched compute, with
GPU-hours reported, to avoid unnecessary resource use. Interpretability analyses
show which interactions drive predictions. All code, data splits and trained
models will be released openly. Outputs are research tools, and their
limitations will be clearly stated.
