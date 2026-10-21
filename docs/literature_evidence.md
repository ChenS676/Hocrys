# Literature evidence: is this a real open problem?

Summary: the claim that crystal graphs are limited to pairwise descriptors is
made explicitly in the literature, and the empirical evidence for explicit
higher-order structure is mixed and confounded.

## The claim being tested is made

- Crystal hypergraph work (CHGCNN, Heilman, Gong & Yan, arXiv:2411.12616) states that
  crystal graphs "can only include pair-wise descriptors" and motivates hyperedges on that basis.
- Counter-position: Peixoto, Peel, Gross & De Domenico, "Graphs are maximally expressive
  for higher-order interactions", arXiv:2602.16937.

## Evidence that higher-order information helps

- **CHGCNN.** Materials Project formation energy: bond-only 0.177 eV/atom vs
  bond+motif 0.074. Matbench phonons: bond-only 84.1 cm^-1 vs bond+motif 64.5.
  Triplets help most on shear modulus (0.099 -> 0.088 log10 GPa).
- **MACE.** With 4-body messages, two layers suffice, whereas two-body message
  passing needs 4-6 layers for comparable accuracy: body order trades against depth.

## Evidence that it often does not matter

- **ALIGNN's own ablation** (JARVIS infrastructure paper, arXiv:2305.11842): removing
  the line graph changed accuracy by more than 10% on only 2 of 13 property tasks,
  both phonon-related (phonon internal energy, heat capacity).
- **CHGCNN.** One motif hyperedge per atom was about as informative as triplets on most
  tasks at much lower cost; combining motifs and triplets diverged or gave no gain.
  For band gap, motif-only was worse than bond-only (0.387 vs 0.315 eV).
- **ALIGNN 2.0** (arXiv:2609.19487, crystal structure prediction): the line graph helps
  denoising, but an additional angle representation gives small, inconsistent effects.

**Consistent pattern:** properties tied to lattice bending and vibration (phonons,
shear modulus) benefit from angular/higher-order information; energies and band
gaps mostly do not.

## Why the evidence does not settle the question

1. **Information vs structure.** Hyperedge models add new geometric features (angles,
   coordination-shape descriptors) absent from the distance-only baseline. An
   equivariant plain graph (e.g. MACE) already resolves the motivating example
   (two local geometries with identical distance graphs).
2. **Weak baselines.** Comparisons are usually against CGCNN (distances only), not a
   plain graph with a rich multivariate interaction function.
3. **No compute matching, weak statistics.** E.g. CHGCNN: single 80/20 split, no
   repeated seeds, no parameter/FLOP matching.
4. **Depth confound.** Body order and depth substitute for each other (MACE result).

## Caveat for the many-body probe

In ACE-type models the internal nu-blocks do not correspond to true body orders
unless the basis is "purified": the two-body component of a model that fits Si
dimers perfectly need not resemble the dimer energy (prediction-rigidity analysis,
arXiv:2408.14311). Read body order only from evaluations on isolated clusters,
and treat it as a diagnostic.

## References

- Peixoto, Peel, Gross, De Domenico. arXiv:2602.16937.
- Arenas, Battiston, Gabrielli (perspective). arXiv:2609.20039.
- "Beyond expressiveness in pairwise and higher-order models". arXiv:2609.26031.
- Heilman, Gong, Yan. Crystal Hypergraph Convolutional Networks. arXiv:2411.12616.
- Choudhary & DeCost. ALIGNN. npj Comput. Mater. 7, 185 (2021).
- Choudhary et al. Recent progress in the JARVIS infrastructure. arXiv:2305.11842.
- ALIGNN 2.0. arXiv:2609.19487.
- Batatia et al. MACE. NeurIPS 2022.
- Prediction rigidities for data-driven chemistry. arXiv:2408.14311.
