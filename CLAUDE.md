# GLARE — Project Overview

## Goal

Gauge-equivariant neural network in Julia/Flux.jl: SU(3) gauge config → rest-eigen
vector correlator C(t) per config, for **cost reduction** in lattice QCD ensembles.

Blueprint: arXiv:2602.21617 (supervised regression) + arXiv:2012.12901 (L-CNN architecture).

## Cost-reduction estimator (central scientific goal)

**This is NOT variance reduction — it is cost reduction.** The estimator is:

```
C_corrected(t) = <C_approx_test(t)> + (<C_exact_bc(t)> - <C_approx_bc(t)>)
```

With optimal NN (MSE training), `Var(C_exact - C_approx) = (1 - r²)·Var(C_exact)`.
At mid-t with scalar plaquette inputs r~0.1 → bias correction adds back ~99% variance.
The equivariant L-CNN is required to push r high enough for the scheme to be viable.
`train_ids` and `bc_ids` must never overlap (enforced by `split_configs`).

## Database design

| File | Content | Builder |
|---|---|---|
| `*_gauge_scalar.h5` | `plaq_scalar` Float64[Lt,Ls,Ls,Ls,npls] = [iL4,iL1,iL2,iL3,npls] | `build_gauge_dataset` |
| `*_gauge_links.h5`  | `gauge_links` ComplexF32[6,Lt,Ls,Ls,Ls,ndim] = [6,iL4,iL1,iL2,iL3,ndim] | `build_gauge_link_dataset` |
| `*_corr.h5`         | LMA re correlators, 3 polarizations | `build_corr_dataset` |

All files share the same config id keys. **Three vector polarizations** `"g1-g1"`, `"g2-g2"`, `"g3-g3"` are training targets; `"g5-g5"` (pseudoscalar) is not.

Data scale: gauge links ~120 MB/config (ComplexF32). 680 configs ≈ 80 GB.
See [src/Dataset/CLAUDE.md](src/Dataset/CLAUDE.md) for schemas and API.

## Repository layout

```
src/
  GLARE.jl                — top-level module
  IO.jl                   — gauge config readers (CERN format)
  Plaquette.jl            — per-site plaquette fields
  Correlator.jl           — LMA correlator reader
  Dataset/                — see src/Dataset/CLAUDE.md
  Preprocessing/          — see src/Preprocessing/CLAUDE.md
  Model/                  — see src/Model/CLAUDE.md
test/                     — see test/CLAUDE.md
main/                     — see main/CLAUDE.md
docs/                     — Documenter.jl source
```

## Phase status

### Phase 0 — Data pipeline ✓
Gauge scalar/link HDF5 builders, correlator builder, 4-way interleaved split,
z-score normalization, `merge_dataset` for server shards.
- [ ] Float32 storage in `build_gauge_dataset`
- [x] **Rebuild `*_gauge_links.h5`** — done (2026-04-15). Old files used `VOL=(48,24,24,24)` (wrong coordinate packing); new database uses correct `VOL=(24,24,24,48)` convention.
- [ ] **Rebuild `*_gauge_scalar.h5`** — still pending. Same VOL bug applies; scalar DB not yet corrected.

### Phase 1 — Baseline CNN ✓ (needs rerun on correct data)
`PeriodicConv4D`, `build_baseline_cnn`, `train_baseline.jl`.
Scalar plaquette ceiling r~0.1 expected even with correct data — coordinate scrambling from
the VOL bug made previous r(t) estimates unreliable. Rebuild scalar DB, then retrain.
Training script updated for GPU (`|> device` pattern, JLD2 checkpointing, TOML config logging).
- [ ] Rebuild `*_gauge_scalar.h5` (see Phase 0 note above) — in progress (2026-04-16)
- [ ] Run full GPU training on correct data, evaluate r(t) on test set

### Phase 2 — Gauge-equivariant L-CNN ✓ (architecture complete, training in progress)
`su3_reconstruct`, `plaquette_matrices`, `ScalarGate`, `TracePool`, `BilinearLayer`,
`GaugeEquivConv`, `LCBBlock`, `build_lcnn` / `LCNN`. All layers gauge-equivariant under
site-dependent V(x). W₀ = plaquette matrices (C_in=6); raw links are NOT valid W₀.
- [x] Rebuild `*_gauge_links.h5` — done (2026-04-15)
- [ ] Smeared inputs (Stout, ρ=0.1) as additional channels (arXiv:2304.10438 §V)
- [x] GPU support — `BilinearLayer{A}` and `GaugeEquivConv{A}` parametrised so `Flux.gpu(model)`
  works correctly; `|> device` pattern in both training scripts; `opt_state` set up after
  `model |> device`; `Flux.cpu(pred)` in `evaluate`; `Float64(loss_val)` for GPU scalar accumulation.
  **CUDA module loading**: CUDA.jl ships its own runtime — `module load system/CUDA` on the
  cluster can override `libcublasLt.so` and break large matmuls (small ones work, large ones
  get `CUBLAS_STATUS_NOT_INITIALIZED`). Fix: use a compatible module version or strip CUDA
  paths from `LD_LIBRARY_PATH` before launching Julia.

### Phase 3 — Training and evaluation
- [x] `lcnn_training.jl` training script (spatial crop for CPU, plaquette W₀)
- [x] Model checkpointing: `lcnn_best.jld2` (best val-loss) and `lcnn_final.jld2` (last epoch),
  saved via JLD2 as `Flux.cpu(model)` for device-agnostic portability.
- [x] Run config logging: `lcnn_config.toml` written at start of every run with all lattice
  dimensions, hyperparameters, split sizes, parameter count, and output paths.
- [x] Gradient accumulation: `ACCUM_STEPS` mini-batches accumulated before one optimizer step.
  Effective batch = `BATCH_SIZE × ACCUM_STEPS`. Keeps peak GPU memory at one config while
  simulating larger batches. Uses `_add_grads`/`_scale_grads` helpers in `lcnn_training.jl`.
- [x] In-memory data preloading: `LINKS_CACHE` (`Dict{String, Array{ComplexF32,7}}`) and
  `CORR_CACHE` (`Dict{String, Matrix{Float32}}`) populated at startup from HDF5 (opened once).
  `su3_reconstruct` called once per config at preload time. Training loop is then pure
  compute — zero HDF5 I/O per step. Requires ~191 MB/config CPU RAM (feasible at 1.6 TB).
- [x] TimerOutputs profiling: `GLARE_TIMER` (global `TimerOutput`) exported from `GLARE.jl`.
  Source-level `@timeit` on data loading / normalization functions. Training script times
  preloading, `plaquette_matrices`, forward+backward, optimizer step, and evaluation.
  `profile_forward(model, U)` runs one AD-free forward pass with per-layer breakdown.
  Full timer printed at end of training run.
- [ ] LR schedule (cosine annealing or ReduceLROnPlateau)
- [ ] r²(t) as primary metric; bias correction at inference
- [ ] Alternative losses: time-weighted MSE or `L = Σ_t (1 - r(t)²)`

### Phase 4 — Extensions
- [ ] Larger Wilson loops (1×2 rectangles) as additional input channels
- [ ] Gauge-equivariant pooling / multiscale U-Net
- [ ] Generalisation across β and L

## Critical implementation notes

- **`_circular_pad` / `_roll` must use `cat`-based slicing** — fancy integer-vector
  indexing silently returns zero gradients through conv layers under Zygote.
- **Baseline CNN: use batch ≥ 32, LR ≤ 1e-3.** Signal is r~0.1 → only 0.25% MSE
  improvement over mean prediction. Batch=8 buries the gradient SNR completely.
- **Float32 training for L-CNN.** Links stored as ComplexF32; unitarity holds to ~1e-6
  after reconstruction. Float32 gives 2-4× GPU throughput via tensor cores.
- **CUDA is a weak dependency only.** `CUDA` declares `__precompile__(false)` — do not
  add as a hard dep. Add as a package extension when GPU kernels are implemented.
- **L-CNN tensor layout:** `(3, 3, Lt, Ls, Ls, Ls, C, B)` — matrix indices first.
- **BilinearLayer uses two-step batched matmul contraction.** Step 1: contract `α` with `W'`
  via a single matrix multiply `(C_out*C_in1, C_in2) × (C_in2, 9N)`. Step 2: contract
  `W` with the result via `batched_mul` over fused `(c,j)` index. This gives O(1) Zygote
  nodes instead of the O(C_in1×C_in2) from a `sum()` generator. The old `sum()` approach
  stored `C_in1×C_in2` arrays of shape `(3,3,C_out,N)` on the tape — catastrophic for
  full volume (e.g. 64 × 9 × 16 × 663K × 8 bytes ≈ 49 GB).
- **W₀ must be plaquette matrices, not raw links.** Links transform as `V(x) U_μ(x) V†(x+μ̂)`
  (different sites on left/right) — NOT gauge-covariant. Use `plaquette_matrices(U_batch)`
  to get `P_μν(x) → V(x) P V†(x)` (C_in=6). Passing links as W₀ silently breaks all
  downstream equivariance and prevents the model from learning.
- **`plaquette_matrices` must not use `push!`.** It can be called inside `withgradient`
  (e.g. user code), so the implementation must remain AD-safe. Use explicit
  `cat(_plane(4,1), ..., _plane(2,1); dims=7)` — no mutation. A `push!(planes, ...)` loop
  triggers "Mutating arrays not supported" under Zygote.
  In `lcnn_training.jl` it is deliberately computed **outside** `withgradient` as
  `W0 = plaquette_matrices(U_batch)` — W₀ is a fixed input feature, not a model parameter,
  so no gradient through the plaquette computation is needed. This avoids Zygote taping
  through it, saves tape memory, and allows separate `@timeit` measurement.
- **GaugeEquivConv uses `Zygote.Buffer` for transport stacking.** All `_pt_fwd`/`_pt_bwd`
  results are written to a pre-sized `Buffer(W, 3, 3, n_ch, N)`, then `copy(buf)` gives
  `PT_all`. This is O(C_in×ndim×N) memory — linear. Do NOT use sequential `cat` in a
  for-loop (`PT_all = cat(PT_all, new; dims=3)`): Zygote stores every growing intermediate,
  giving O((C_in×ndim)² × N) — quadratic and catastrophic for full volume. Array
  comprehensions (`[f(j,mu) for j in ..., mu in ...]`) also fail: `push!` internally.
  `Zygote.Buffer` is the only Zygote-safe pattern for building an array of runtime-determined
  size from a loop.
- **`plaquette_matrices` direction convention:** LatticeGPU uses `1=x, 2=y, 3=z, 4=t`.
  Array layout is `(3,3,Lt,Ls,Ls,Ls,B)` with `dim 3=t, dim 4=x, dim 5=y, dim 6=z`.
  The direction→dim map is `(4,5,6,3)[mu]` — direction 4(t)→dim3, 1(x)→dim4, 2(y)→dim5, 3(z)→dim6.
  This differs from the naive `mu+2` because `_to_spatial_links` places the temporal coordinate
  (coord4, period `iL[4]=Lt`) at dim1 of its output — so after `su3_reconstruct` adds 2 matrix
  dims, temporal lands at dim3. `VOL=(Lx,Ly,Lz,Lt)=(24,24,24,48)` with `iL[4]=48=Lt`.
  Same map applies in `GaugeEquivConv`. Plane ordering `(4,1),(4,2),(4,3),(3,1),(3,2),(2,1)`
  matches `plaquette_field` / Plaquette.jl.
- **HDF5 metadata `vol` convention:** `_write_gauge_metadata` stores `vol = collect(lp.iL)`.
  With correct LatticeGPU convention `lp.iL = (Lx, Ly, Lz, Lt)`, time is at index 4.
  Always read `Lt = vol[4]`, `Ls = vol[1]`. **Never `vol[1]` for `Lt`** — old databases built
  with the wrong `VOL=(48,24,24,24)` had `vol[1]=48=Lt` coincidentally, masking this bug.
- **Two-level gradient checkpointing in LCNN/LCBBlock.**
  - *Block level* (`LCNN`): `Zygote.checkpointed(blk, W, U)` wraps each `LCBBlock` —
    stores only the block input W on the tape, reruns the block forward during backward.
    Eliminates inter-block tape.
  - *Sub-layer level* (`LCBBlock`): `Zygote.checkpointed(l.conv, W, U)` wraps
    `GaugeEquivConv` inside each block. Keeps the conv's `Zygote.Buffer` intermediates
    off the tape while `BilinearLayer` backward runs, reducing peak GPU memory.
  - Cost: conv runs 3x total (1x forward, 1x block rerun, 1x inner rerun during block
    rerun's backward). ~50% more compute per step.
  - Full 24³×48 training fits on 80 GB GPU with `channels=[8,16]` (peak ~15 GB per block).
  - `TRAIN_CROP_S = Ls` (full volume) is the default; set to e.g. 16 to re-enable
    cropping if checkpointing is unavailable. `random_spatial_crop` kept as fallback.
- **LCBBlock:** `BilinearLayer(W_local, W_transported)` — one-link loops at first block;
  each stacked block doubles Wilson loop extent.
- **Normalization:** corr stats on source-averaged `C̄(t)`, not per-source. Per-config
  normalization would destroy the config-to-config fluctuations that are the signal.
  Use `compute_normalization` (Phase 1, needs gauge HDF5) or `compute_corr_normalization`
  (Phase 2, correlator only — `feat_*` fields are empty to prevent accidental misuse).

See [src/CLAUDE.md](src/CLAUDE.md) for LatticeGPU quirks and quick reference.

## Documentation

Site: https://bunniies.github.io/GLARE.jl

```bash
julia docs/make.jl
cd docs/build && git init && git add -A && git commit -m "Deploy docs" \
  && git push --force git@github.com:Bunniies/GLARE.jl.git HEAD:gh-pages && cd ../..
```

CI cannot deploy automatically (LatticeGPU/BDIO on private server). Always deploy locally.

## Key paper references

| arXiv | Role |
|---|---|
| 2602.21617 | Application blueprint: config-by-config regression + bias correction |
| 2012.12901 | Foundational L-CNN: parallel transport conv, scalar gate, trace layer |
| 2304.10438 | Gauge-equivariant pooling + smeared inputs as multi-scale features |
| 2602.23840 | Novel gauge-equivariant architecture (Pfahler et al. 2026) |
| 2501.16955 | Gauge-covariant Transformer — future extension |
