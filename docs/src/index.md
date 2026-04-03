# GLARE.jl

**Gauge-equivariant neural network for lattice QCD cost reduction.**

GLARE predicts rest-eigen vector correlators $C(t)$ from SU(3) gauge configurations
using a gauge-equivariant L-CNN architecture. The goal is **cost reduction** in
lattice QCD ensembles: cheap per-config NN predictions replace expensive LMA
computations on the bulk of the ensemble, with a small bias-correction subset
restoring exactness.

## Cost-reduction estimator

```math
C_\text{corrected}(t) = \langle C_\text{approx}^\text{test}(t) \rangle
  + \left(\langle C_\text{exact}^{bc}(t) \rangle - \langle C_\text{approx}^{bc}(t) \rangle\right)
```

- ``C_\text{approx}(t)`` — NN prediction from gauge config (cheap, computed for every config)
- ``C_\text{exact}(t)``  — full LMA correlator (expensive, only for the bias-correction subset)

With optimal NN (MSE training), ``\text{Var}(C_\text{exact} - C_\text{approx}) = (1-r^2)\,\text{Var}(C_\text{exact})``.
The equivariant L-CNN is required to push $r$ high enough for the scheme to be viable.

## Workflow

```
Phase 0 — Data pipeline
  build_gauge_dataset / build_gauge_matrix_dataset / build_corr_dataset
  → merge_dataset (merge per-range server shards)
  → split_configs + compute_normalization

Phase 1 — Baseline CNN  (scalar plaquette input)
  build_baseline_cnn → train → evaluate r(t)

Phase 2 — L-CNN  (SU(3) matrix input, gauge-equivariant)
```

## References

| arXiv | Role |
|---|---|
| [2602.21617](https://arxiv.org/abs/2602.21617) | Application blueprint: config-by-config regression + bias correction |
| [2003.06413](https://arxiv.org/abs/2003.06413) | Foundational L-CNN: parallel transport conv, scalar gate, trace layer |
| [2304.10438](https://arxiv.org/abs/2304.10438) | Gauge-equivariant pooling + smeared inputs |

## Index

```@index
```
