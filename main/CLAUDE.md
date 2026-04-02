# main/ — Scripts Reference

## Scripts

| Script | Purpose |
|---|---|
| `check_dataset.jl` | Build gauge scalar + correlator HDF5, inspect structure |
| `check_normalization.jl` | Normalization checks + Pearson r plots (PyPlot) |
| `train_baseline.jl` | Phase 1 baseline CNN training loop |

## Environment variables

| ENV var | Points to |
|---|---|
| `GLARE_TEST_GAUGE_H5` | `A654_gauge_scalar.h5` — scalar gauge HDF5 |
| `GLARE_TEST_MATRIX_H5` | `A654_gauge_matrix.h5` — matrix gauge HDF5 (Phase 2) |
| `GLARE_TEST_CORR_H5` | `A654_corr.h5` — correlator HDF5 |
| `GLARE_PLOT_DIR` | Output dir for diagnostic PDFs |

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
VOL   = (48, 24, 24, 24)   # (T, Lx, Ly, Lz)
SVOL  = (8, 4, 4, 4)
BC    = BC_PERIODIC
TWIST = (0, 0, 0, 0, 0, 0)
lp    = SpaceParm{4}(VOL, SVOL, BC, TWIST)
```

## Recommended server workflow

Gauge configs and LMA data live on an external server. Recommended pipeline:
1. Copy GLARE to server
2. `build_gauge_dataset(ensemble_path, lp, scalar_h5; config_range=1:100)` — one job per range
3. `build_corr_dataset(lma_path, corr_h5; config_range=1:100)`
4. Merge per-range HDF5 files locally (`merge_dataset` — not yet implemented)
5. All training runs locally on merged files

Per-range HDF5 files avoid file-lock contention in SLURM parallel jobs.
