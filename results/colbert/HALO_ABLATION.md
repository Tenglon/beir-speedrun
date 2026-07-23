# Hyperbolic (HALO-style) ColBERT: a seven-configuration failure taxonomy

Goal: LoRA bert-base + hyperboloid lift head (learnable curvature, init 0.1) +
HALO similarity, trained by listwise KD on ms-marco-en-bge-gemma, matching the
euclidean ColBERT baseline (small6 avg nDCG@10 0.464 / full BEIR 0.5085).

Small6 avg nDCG@10 per configuration (10k/20k/30k/60k steps where measured):

| cfg | change vs previous | 10k | 20k | 30k | 60k | failure mode |
|---|---|---:|---:|---:|---:|---|
| run3 | pure −d_L, KD only | 0.032 | 0.036 | 0.044 | 0.049 | vertex collapse: softmax is shift/scale-blind, microscopic distances fit the teacher |
| run4 | + hybrid 0.5cos (HALO mainline) | 0.063 | 0.058 | — | — | collapses anyway: cosine KD branch is equally scale-blind (spatial norms 2.5→0.77) |
| run5 | + z-score KD, + InfoNCE (128 local negs) | 0.223 | — | 0.233 | 0.233 | no collapse, but quality decays with corpus size (trec-covid 5% of euclidean): local negatives shape only neighborhood structure |
| iterA | + cross-GPU gather (2048 negs) | 0.022 | — | — | — | diverges mid-run (grad norms 750+): acosh's 1/√(z²−1) singularity × CE pulling positives to d=0 × 16x more near-duplicate pairs |
| iterB | + squared distance −d² (finite gradient) | 0.284 | — | — | — | best honest point — but partly an lr-annealing artifact (schedule ends at 10k) |
| iterC | iterB config, 30k schedule | 0.198 | 0.082 | 0.086 | — | with lr still active, the degenerate drift reasserts; long queries also hurt by −d²'s quadratic penalty in the MaxSim sum (arguana halves) |
| iterD | + norm squash (R≈4) + entailment hinge λ0.2 | 0.004 | 0.002 | 0.002 | — | norm saturation: NCE rewards larger radii → all tokens pinned at the squash ceiling (norms 3.700x, query max=mean), squash Jacobian ≈ R/n² throttles backbone gradients 50x; uniform radius reduces Lorentz distance to a monotone function of cosine |

## Core findings

1. **Listwise KD is shift- and scale-invariant per candidate list; hyperbolic
   scale is a free variable.** Every un-anchored configuration found a
   degenerate optimum (vertex collapse, saturation) that satisfies the
   objective while destroying global retrieval structure. The euclidean
   baseline is protected by accident: pylate's min-max normalization makes the
   loss scale-invariant, so there is no gradient pressure on scale at all.
2. **Each guard removed one degenerate channel and revealed the next**
   (z-score → local-negative myopia → acosh singularity → annealing mirage →
   saturation). This whack-a-mole pattern is the signature of an objective
   that underdetermines the geometry's degrees of freedom, not of bad luck.
3. **Uniform token radius makes the hyperboloid pointless**: with equal norms,
   −d_L is rank-equivalent to cosine per token pair. Norm freedom is both the
   carrier of hyperbolic information (abstraction/hierarchy) and the collapse
   channel; it needs an explicit, well-posed job, not just a hinge.

## What we would try next (untried, structurally different)

- Curriculum from the working euclidean checkpoint (run2, 0.5085): introduce
  the lift on an already-organized space with frozen-then-annealed curvature,
  instead of fighting degenerate attractors from random init.
- Entailment-cone losses (angle-based, MERU/HyCoCLIP-style) instead of raw
  norm hinges; margin-MSE pairwise KD (scale-sensitive) instead of listwise KL.
- Radial structure analysis (token type vs distance-to-vertex) as an
  acceptance test for whether the manifold is genuinely used.

Infrastructure delivered regardless: HALOLift ST module (learnable curvature /
temperatures, squash, save/load), Lorentz + hybrid sharded MaxSim retrieval at
33.9M-doc scale, LoRA-checkpoint converter, convert-then-eval automation.
Total campaign cost ≈ 45 GPU·h across 7 configurations.
