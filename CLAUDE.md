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
- **Data scale:** plaq_scalar ~30 MB/config (Float64). 680 configs ≈ 21 GB.
  Before full-scale training: convert to Float32 (~10 GB) and preload into RAM.

## Repository layout

```
src/
  GLARE.jl          — top-level module
  IO.jl             — gauge config readers (CERN format)
  Plaquette.jl      — per-site plaquette fields (untraced + scalar)
  Correlator.jl     — LMA correlator reader (rest-eigen only)
  Dataset.jl        — build_gauge_dataset, build_gauge_matrix_dataset, build_corr_dataset, merge_dataset
  Preprocessing.jl  — split, normalization, data loading
  Model.jl          — PeriodicConv4D, build_baseline_cnn, pearson_r
test/               — see test/CLAUDE.md
main/               — see main/CLAUDE.md
docs/               — Documenter.jl source (make.jl, src/)
PLAN.md             — full phased implementation plan
```

See `src/CLAUDE.md` for per-module API details.

## What's done / what's next

### Phase 0 — Data pipeline ✓
- Gauge scalar + matrix HDF5 builders with `config_range` for parallel server jobs
- Correlator HDF5 builder (3 polarizations)
- 4-way interleaved split (train/val/test/bc), global z-score normalization
- `merge_dataset` — merges per-range server shards into a single file
- Diagnostic scripts: `check_dataset.jl`, `check_normalization.jl`
- Pearson r analysis: scalar plaquette gives r~0.1 at mid-t → equivariant CNN required

### Phase 0 (remaining)
- [ ] Float32 storage in `build_gauge_dataset` — halves disk/RAM usage (see data scale note above)

### Phase 1 — Baseline CNN ✓ (implemented, training in progress)
- [x] `PeriodicConv4D` — 3D spatial conv with Lt folded into batch (NNlib has no 4D conv support)
- [x] `build_baseline_cnn` — full chain: conv → spatial mean → MLP → `(Lt, npol, B)`
- [x] `train_baseline.jl` — training loop, Adam, per-epoch val loss + Pearson r, CSV log + plots
- [x] Normalization bug fixed: corr stats computed on source-averaged targets (not per-source)
- [x] `PreloadedDataset` / `preload_dataset` — in-memory cache, eliminates per-batch HDF5 reads; `USE_PRELOAD` flag in training script keeps per-batch HDF5 path available
- [x] `_circular_pad` rewritten with `cat`-based slicing — fixes silent Zygote AD breakage from fancy indexing; confirmed gradients flow to both conv layers (norms ~1e-3)
- [x] Training hyperparameters fixed: LR=1e-3 (was 5e-3), batch=32 (was 8), epochs=300 —
      original run showed flat loss (~1.0) for 43 epochs due to gradient SNR too low for the
      weak scalar-plaquette signal (r~0.1 → only ~0.25% MSE improvement; batch=8 insufficient)
- [ ] Run full training on complete dataset, evaluate r(t) on test set

**Scientific note:** scalar plaquette ceiling is r~0.1, making (1 - r²) ≈ 0.99 — the bias
correction scheme adds back ~99% of variance at this r, which is not viable for cost reduction.
The baseline CNN is an infrastructure/lower-bound check only. Phase 2 (L-CNN on SU(3) matrices)
is required to reach r large enough for the estimator to be useful.

### Phase 2 — Gauge-equivariant L-CNN
- [x] `su3_reconstruct`: recover full 3×3 SU(3) from first-two-row storage `(6,...) → (3,3,...)`
- [ ] `ScalarGate`: `σ(Re(Tr Φ)) * Φ` nonlinearity
- [ ] `TracePool`: Re(Tr Φ) → spatial mean → `(Lt, n_ch, B)`
- [ ] `GaugeEquivConv`: parallel transport + scalar channel weights (arXiv:2003.06413) — requires link matrices
- [ ] `BilinearLayer`: `Φ_out^a = Σ_{b,c} W_{abc} Φ^b · Φ^c`
- [ ] Gauge-equivariance unit test
- [ ] Build `*_gauge_matrix.h5` for Phase 2 inputs (builder already implemented, needs to be run)
- [ ] Extend dataset to store raw link matrices `U_μ(x)` for parallel transport in `GaugeEquivConv`
- [ ] Smeared input copies (Stout, ρ=0.1) as additional channels (arXiv:2304.10438 §V)
- [ ] GPU support (CuArray-compatible kernels) — baseline CNN too small to benefit

### Phase 3 — Training and evaluation
- [ ] LR schedule (cosine annealing or ReduceLROnPlateau)
- [ ] Variance reduction factor r²(t) as primary metric
- [ ] Bias correction: `C_corrected(t) = C_pred(t) + mean_bc(C_true - C_pred)(t)`
- [ ] Alternative loss functions: time-weighted MSE or direct r-maximizing `L = Σ_t (1 - r(t)²)`

### Phase 4 — Extensions
- [ ] Larger Wilson loops (1×2 rectangles) as additional input channels
- [ ] Gauge-equivariant pooling / multiscale U-Net
- [ ] Generalisation across β and L

## Known implementation notes

- **NNlib has no 4D conv support**: `PeriodicConv4D` uses a 3D spatial conv with
  `Lt` folded into the batch dimension. Architecturally equivalent for the baseline.
- **`_circular_pad` must use `cat`-based slicing**, not fancy integer-vector indexing
  (`x[idx1, idx2, idx3, :, :]`). The latter silently returns zero gradients through
  the conv layers under Zygote. Confirmed fix: grad norms conv1~1.8e-3, conv2~8.3e-3.
- **Baseline CNN training requires large batch + low LR**: the scalar plaquette signal is
  r~0.1, meaning optimal MSE is only ~0.25% below the "predict mean" baseline. Batch=8 gives
  gradient SNR ≈ 0.28, completely burying the signal. Use batch ≥ 32 and LR ≤ 1e-3 with Adam.
  Memory note: each conv layer processes `(26³ × 48 × B)` Float32 tensors; batch=32 ≈ 4-6 GB
  Zygote tape, batch=64 ≈ 12 GB.
- **CUDA removed from deps**: `CUDA` declares `__precompile__(false)` and breaks
  precompilation of the entire package. Do not re-add until GPU kernels are implemented
  (Phase 2+), and add it as a weak dependency with an extension, not a hard dep.
- **Normalization**: `compute_normalization` computes corr stats on source-averaged
  `C̄(t)` — not per-source. This matches the training target and gives unit-variance
  normalized targets.
- **Training target**: source-averaged correlator per polarization `→ (Lt, npol)`.
  Network cannot predict within-config source fluctuations; source-averaging before
  training is optimal.

## Documentation

Site: https://bunniies.github.io/GLARE.jl

To update after changing docstrings or markdown pages:

```bash
julia docs/make.jl
cd docs/build && git init && git add -A && git commit -m "Deploy docs" && git push --force git@github.com:Bunniies/GLARE.jl.git HEAD:gh-pages && cd ../..
```

CI cannot deploy automatically because LatticeGPU/BDIO are on a private server
(igit.ific.uv.es) unreachable from GitHub Actions. Always deploy locally.

## Key paper references

| arXiv | Role |
|---|---|
| 2602.21617 | Application blueprint: config-by-config regression + bias correction |
| 2003.06413 | Foundational L-CNN: parallel transport conv, scalar gate, trace layer |
| 2304.10438 | Gauge-equivariant pooling + smeared inputs as multi-scale features |
| 2602.23840 | Novel gauge-equivariant architecture (Pfahler et al. 2026) |
| 2501.16955 | Gauge-covariant Transformer — future extension |
