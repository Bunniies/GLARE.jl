# main/ — Scripts Reference

## Scripts

| Script | Purpose |
|---|---|
| `baseline_training/check_dataset.jl` | Build gauge scalar + correlator HDF5, inspect structure |
| `baseline_training/check_normalization.jl` | Normalization checks + Pearson r plots (PyPlot) |
| `baseline_training/train_baseline.jl` | Phase 1 baseline CNN training loop |
| `LCNN_training/lcnn_training.jl` | Phase 2/3 L-CNN training loop (gauge links HDF5 + correlator HDF5) |

## Environment variables

| ENV var | Points to |
|---|---|
| `GLARE_TEST_GAUGE_H5` | `A654_gauge_scalar.h5` — scalar gauge HDF5 |
| `GLARE_TEST_MATRIX_H5` | `A654_gauge_matrix.h5` — matrix gauge HDF5 (Phase 2) |
| `GLARE_TEST_CORR_H5` | `A654_corr.h5` — correlator HDF5 |
| `GLARE_LOG_DIR` | Output dir for L-CNN training logs and plots |
| `GLARE_PLOT_DIR` | Output dir for diagnostic PDFs |

## L-CNN training notes (`LCNN_training/lcnn_training.jl`)

- **W₀ = `plaquette_matrices(U_batch)`** (C_in=6). Never pass raw links as W₀ — see top-level CLAUDE.md.
- **Spatial crop** (`TRAIN_CROP_S`): defaults to `Ls` (full volume) now that gradient checkpointing is enabled in `LCNN`. Set to e.g. 16 to re-enable cropping as a fallback if checkpointing causes issues. `random_spatial_crop` is kept in the script for this purpose.
- **`model(plaquette_matrices(U_batch), U_batch)`** — plaquettes as W₀, links as transport field U.
- **GPU support**: `const device = Flux.gpu_device()` at top; `model |> device`, `opt_state`
  set up after; `U_batch |> device` and `corr |> device` in training loop; `Flux.cpu(pred)`
  before metric accumulation; `Float64(loss_val)` to extract scalar from GPU.
- **Checkpointing**: `lcnn_best.jld2` saved whenever val-loss improves; `lcnn_final.jld2`
  saved unconditionally after last epoch. Both store `Flux.cpu(model)` for portability.
  Load with `JLD2.load(path)["model"] |> device`.
- **Run config**: `lcnn_config.toml` written to `LOG_DIR` at the start of every run.
  Contains lattice dims, split sizes, all hyperparameters, parameter count, device, and
  output paths. Useful for reproducing runs and post-hoc debugging.
- **`vol` metadata reading**: `_write_gauge_metadata` stores `lp.iL = (Lx, Ly, Lz, Lt)`.
  Read as `Lt = vol[4]`, `Ls = vol[1]`. See top-level CLAUDE.md for why `vol[1]` is wrong.

## Local data paths (CLS A654)

```
Lattice/data/cls/                          ← gauge configs (~800 MB each)
Lattice/data/HVP/LMA/A654_all_t_sources/
  dat/                                     ← LMA source data (per-config integer dirs)
  hdf5/A654_all_t_sources/
    A654_gauge_scalar.h5                   ← scalar plaq features, spatial layout
    A654_gauge_matrix.h5                   ← SU3 matrix features (Phase 2, build separately)
    A654_corr.h5                           ← LMA correlators, 3 polarizations
    plots/                                 ← diagnostic PDFs from check_normalization.jl
```

## CLS A654 lattice parameters

```julia
VOL   = (24, 24, 24, 48)   # (Lx, Ly, Lz, Lt) — LatticeGPU convention: t at index 4
SVOL  = (4, 4, 4, 8)
BC    = BC_PERIODIC
TWIST = (0, 0, 0, 0, 0, 0)
lp    = SpaceParm{4}(VOL, SVOL, BC, TWIST)
```

**Old wrong convention was `VOL=(48,24,24,24)` (t first).** This gave `lp.iL[4]=24` so only
24 t-slices were read from CERN files. The training scripts read `Lt = vol[4]`, `Ls = vol[1]`
from HDF5 metadata — this is correct only for databases built with the right convention above.

## Recommended server workflow

Gauge configs and LMA data live on an external server. Recommended pipeline:
1. Copy GLARE to server
2. `build_gauge_dataset(ensemble_path, lp, scalar_h5; config_range=1:100)` — one job per range
3. `build_corr_dataset(lma_path, corr_h5; config_range=1:100)`
4. Merge per-range HDF5 files locally (`merge_dataset` — not yet implemented)
5. All training runs locally on merged files

Per-range HDF5 files avoid file-lock contention in SLURM parallel jobs.
