# src/Preprocessing/ — Preprocessing module

Splits configs, computes normalization statistics, and loads data for training.
Split into two files plus the module entry point.

## File layout

```
Preprocessing.jl  — module entry, exports
Normalization.jl  — split_configs, NormStats, compute/save/load_normalization
DataLoading.jl    — load_*, PreloadedDataset, preload_dataset
```

## Public API

### Normalization.jl

**`split_configs(h5path; train=0.60, val=0.15, test=0.15, bias_corr=0.10)`**
→ `(train_ids, val_ids, test_ids, bc_ids)` as `Vector{String}` in MC order.
Interleaved Bresenham assignment — preserves chain order, maximises separation.
`train_ids ∩ bc_ids = ∅` by construction.

**`NormStats`** — struct with `feat_mean`, `feat_std` (length npls) and `corr_mean`, `corr_std` (length T).

**`compute_normalization(gauge_h5, corr_h5, train_ids; polarizations)`** → `NormStats`
Phase 1 (baseline CNN). Computes both gauge feature stats (`feat_mean`/`feat_std` from `plaq_scalar`)
and correlator stats. Stats computed on source-averaged `C̄(t)`. Never touches val/test/bc data.

**`compute_corr_normalization(corr_h5, train_ids; polarizations)`** → `NormStats`
Phase 2 (L-CNN). Correlator stats only — no gauge file needed since links are not normalized.
Returns `NormStats` with `feat_mean = Float64[]`, `feat_std = Float64[]` (empty by design —
passing to any gauge normalization function will error rather than silently misbehave).

**`save_normalization(h5path, stats)`** / **`load_normalization(h5path)`** → `NormStats`
Stored in `normalization/` group of the gauge HDF5 file.

### DataLoading.jl

**`load_gauge(gauge_h5, cid; stats, field=:scalar)`**
- `field=:scalar` → `Float64[iL1,iL2,iL3,iL4,npls]` (from `*_gauge_scalar.h5`)
- `field=:matrix` → `ComplexF64[6,iL1,iL2,iL3,iL4,npls]` (from `*_gauge_matrix.h5`)
- `field=:both` is **not supported** — load from the two databases separately.
- Applies z-score normalization if `stats` provided (scalar only; matrix is never normalized).

**`load_corr(corr_h5, cid; stats, polarization="g1-g1")`** → `Float64[T, nsrcs]`

**`load_config(gauge_h5, corr_h5, cid; stats, field, polarization)`** → `(features, correlator)`

**`load_split(gauge_h5, corr_h5, ids; stats, field, polarization)`** → `(feats_list, corrs_list)`

**`load_links(links_h5, cid)`** → `ComplexF32[6, iL1, iL2, iL3, iL4, ndim]`
Raw first-two-row link storage for one config. No normalization applied.
Call `su3_reconstruct` on the result to get full 3×3 SU(3) matrices.

**`PreloadedDataset`** — in-memory cache `cid → (feat::Float32[iL1,iL2,iL3,iL4,npls], corr2d::Float32[Lt,npol])`.
Build with `preload_dataset`; pass to training loop to avoid per-batch HDF5 reads.

**`preload_dataset(gauge_h5, corr_h5, ids, stats; polarizations)`** → `PreloadedDataset`
Opens each HDF5 file once; stores normalized Float32 arrays. Scalar db only (matrix too large).
Memory: ~24 MB/config on A654 (48×24³×6 Float32). 680 configs ≈ 16 GB.

## Notes

- **Global z-score normalization on train only.** Per-config normalization would destroy
  the config-to-config fluctuations that are the regression signal.
- **Normalization target is source-averaged.** Both `compute_normalization` and
  `compute_corr_normalization` operate on `C̄(t) = mean(C[:,t])`, not raw per-source values,
  matching what the network is trained to predict.
- **Phase 1 vs Phase 2 normalization:** use `compute_normalization` for the baseline CNN
  (needs gauge feature stats); use `compute_corr_normalization` for the L-CNN (links are not
  normalized — empty `feat_*` fields in `NormStats` guard against accidental misuse).
