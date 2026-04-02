# GLARE — Project Overview

## Goal

Gauge-equivariant neural network in Julia/Flux.jl: SU(3) gauge config → rest-eigen
vector correlator C(t) per config, for **cost reduction** in lattice QCD ensembles.

Blueprint: arXiv:2602.21617 (supervised regression) + arXiv:2003.06413 (L-CNN architecture).

## Cost-reduction estimator (central scientific goal)

**This is NOT variance reduction — it is cost reduction.** The estimator is:

```
C_corrected(t) = <C_approx_test(t)> + (<C_exact_bc(t)> - <C_approx_bc(t)>)
```

- `C_approx(t)` = NN prediction from gauge config (cheap, per-config)
- `C_exact(t)`  = full LMA correlator (expensive, only for a subset)
- `test` subset = all configs where only gauge field is available
- `bc` subset   = small set with both gauge field AND exact LMA (bias correction)

With optimal NN (MSE training), `Var(C_exact - C_approx) = (1 - r²)·Var(C_exact)`.
At mid-t with scalar plaquette inputs r~0.1 → bias correction adds back ~99% variance.
The equivariant L-CNN is required to push r high enough for the scheme to be viable.

`train_ids` and `bc_ids` must never overlap (enforced by `split_configs`).

## Three-database design

| File | Content | Built by |
|---|---|---|
| `*_gauge_scalar.h5` | `plaq_scalar` Float64[iL[1],iL[2],iL[3],iL[4],npls] | `build_gauge_dataset` |
| `*_gauge_matrix.h5` | `plaq_matrix` ComplexF64[6,iL[1],iL[2],iL[3],iL[4],npls] | `build_gauge_matrix_dataset` |
| `*_corr.h5`         | LMA re correlators, all 3 polarizations | `build_corr_dataset` |

- Scalar and matrix databases are separate files — matrix (~8× larger) only needed for Phase 2.
- All three files share the same config id keys (trailing integer after `n` in gauge filenames).
- **Three vector polarizations:** `"g1-g1"`, `"g2-g2"`, `"g3-g3"` — all used as training targets,
  tripling the training signal. `"g5-g5"` (pseudoscalar) is NOT a target.

## Repository layout

```
src/
  GLARE.jl          — top-level module
  IO.jl             — gauge config readers (CERN format)
  Plaquette.jl      — per-site plaquette fields (untraced + scalar)
  Correlator.jl     — LMA correlator reader (rest-eigen only)
  Dataset.jl        — build_gauge_dataset, build_gauge_matrix_dataset, build_corr_dataset
  Preprocessing.jl  — split, normalization, data loading
  Model.jl          — PeriodicConv4D, build_baseline_cnn, pearson_r
test/               — see test/CLAUDE.md
main/               — see main/CLAUDE.md
PLAN.md             — full phased implementation plan
```

See `src/CLAUDE.md` for per-module API details.

## What's done / what's next

### Phase 0 — Data pipeline ✓
- Gauge scalar + matrix HDF5 builders with `config_range` for parallel server jobs
- Correlator HDF5 builder (3 polarizations)
- 4-way interleaved split (train/val/test/bc), global z-score normalization
- Diagnostic scripts: `check_dataset.jl`, `check_normalization.jl`
- Pearson r analysis: scalar plaquette gives r~0.1 at mid-t → equivariant CNN required

### Phase 0 (remaining)
- [x] `merge_dataset` — merge per-config HDF5 files (from server) into single training file

### Phase 1 — Baseline CNN ✓ (implemented, not yet trained)
- [x] `PeriodicConv4D` — 4D conv with exact circular padding
- [x] `build_baseline_cnn` — full chain: conv → spatial mean → MLP → `(Lt, npol, B)`
- [x] `train_baseline.jl` — training loop, Adam, per-epoch val loss + Pearson r
- [ ] Run training, evaluate r(t) on test set — sets performance floor for L-CNN

### Phase 2 — Gauge-equivariant L-CNN
- [ ] `GaugeEquivConv`: parallel transport + scalar channel weights (arXiv:2003.06413)
- [ ] `ScalarGate`: `σ(Re(Tr Φ)) * Φ` nonlinearity
- [ ] `BilinearLayer`: `Φ_out^a = Σ_{b,c} W_{abc} Φ^b · Φ^c`
- [ ] `TraceAndAggregate`: Re(Tr Φ) → spatial mean → `[Lt, channels]`
- [ ] Gauge-equivariance unit test
- [ ] Build `*_gauge_matrix.h5` for Phase 2 inputs
- [ ] Smeared input copies (Stout, ρ=0.1) as additional channels (arXiv:2304.10438 §V)

### Phase 3 — Training and evaluation
- [ ] Weighted MSE loss (`1/var(C(t))`)
- [ ] LR schedule (cosine annealing or ReduceLROnPlateau)
- [ ] Variance reduction factor r²(t) as primary metric
- [ ] Bias correction: `C_corrected(t) = C_pred(t) + mean_bc(C_true - C_pred)(t)`

### Phase 4 — Extensions
- [ ] Larger Wilson loops (1×2 rectangles) as additional input channels
- [ ] GPU support (CuArray-compatible kernels)
- [ ] Gauge-equivariant pooling / multiscale U-Net
- [ ] Generalisation across β and L

## Key paper references

| arXiv | Role |
|---|---|
| 2602.21617 | Application blueprint: config-by-config regression + bias correction |
| 2003.06413 | Foundational L-CNN: parallel transport conv, scalar gate, trace layer |
| 2304.10438 | Gauge-equivariant pooling + smeared inputs as multi-scale features |
| 2602.23840 | Novel gauge-equivariant architecture (Pfahler et al. 2026) |
| 2501.16955 | Gauge-covariant Transformer — future extension |
