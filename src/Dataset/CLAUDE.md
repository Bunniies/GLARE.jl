# src/Dataset/ — Dataset module

Builds HDF5 databases from raw gauge configs and LMA correlator files.
Split into three files plus the module entry point.

## File layout

```
Dataset.jl       — module entry, shared private helpers, exports
GaugeDataset.jl  — gauge builders + layout converters
CorrDataset.jl   — correlator builder
Merge.jl         — merge_dataset
```

## Public API

**`build_gauge_dataset(ensemble_path, lp, output_path; config_fmt, config_range, verbose)`**
Stores `plaq_scalar = Re(Tr P_μν(x))` as `Float64[Lt,Ls,Ls,Ls,npls]` = `Float64[iL4,iL1,iL2,iL3,npls]`.

**`build_gauge_matrix_dataset(ensemble_path, lp, output_path; config_fmt, config_range, verbose)`**
Stores `plaq_matrix` as `ComplexF64[6,Lt,Ls,Ls,Ls,npls]` — first 2 rows of untraced P_μν(x).
Superseded by `build_gauge_link_dataset` for Phase 2.

**`build_gauge_link_dataset(ensemble_path, lp, output_path; config_fmt, config_range, verbose)`**
Stores `gauge_links` as `ComplexF32[6,Lt,Ls,Ls,Ls,ndim]` = `ComplexF32[6,iL4,iL1,iL2,iL3,ndim]`.
Temporal coordinate placed first: after `su3_reconstruct` the array is `(3,3,Lt,Ls,Ls,Ls,ndim)`.
Written as Float32 (half of Float64; SU(3) unitarity holds to ~1e-6 after reconstruction).
Primary input for Phase 2 L-CNN.

**`build_corr_dataset(lma_path, output_path; em, polarizations, config_range, verbose)`**
Stores all three vector polarizations per config. Default `polarizations=["g1-g1","g2-g2","g3-g3"]`.

**`merge_dataset(input_paths, output_path; verbose)`**
Merges per-range server shards into one file. Works for all database types.
Metadata copied from first shard; `vol`/`svol` consistency checked. Duplicate ids raise an error.

## HDF5 schemas

```
gauge_scalar_db.h5
├── metadata/  vol Int64[4], svol Int64[4], ensemble String, config_fmt String
└── configs/<id>/plaq_scalar   Float64[iL4,iL1,iL2,iL3,npls]   (= [Lt,Ls,Ls,Ls,npls])

gauge_links_db.h5
├── metadata/  vol, svol, ensemble, config_fmt
└── configs/<id>/gauge_links   ComplexF32[6,iL4,iL1,iL2,iL3,ndim]  (= [6,Lt,Ls,Ls,Ls,ndim])

corr_db.h5
├── metadata/  lma_path, em, polarizations
└── configs/<id>/<polarization>/
    ├── correlator  Float64[T, nsrcs]
    └── sources     String[nsrcs]
```

Config ids: trailing integer after `n` in gauge filenames (`A654r000n1` → `"1"`).
`config_range` indexes into the sorted list — use for parallel server shards.

## Shared private helpers (in Dataset.jl)

- `_config_id(filename)` — extracts trailing integer id from gauge filename
- `_gauge_map(ensemble_path)` → `Dict{String,String}` id → filename
- `_gauge_config_ids(ensemble_path, config_range, verbose)` → sorted id list
- `_check_range(r, n)` — bounds check for config_range
- `_write_gauge_metadata(fid, lp, ensemble_path, config_fmt)` — writes vol/svol/etc.
- `_make_reader(fmt, lp)` → reader closure; only `"cern"` supported

## Layout converters (in GaugeDataset.jl)

- `_to_spatial_scalar(ps, lp)` — `(bsz,npls,rsz)` → `Float64[iL4,iL1,iL2,iL3,npls]` (t first)
- `_to_spatial_matrix(pf, lp)` — `(bsz,npls,rsz)` SU3 → `ComplexF64[6,iL4,iL1,iL2,iL3,npls]` (t first)
- `_to_spatial_links(U, lp)`  — `(bsz,ndim,rsz)` SU3 → `ComplexF64[6,iL4,iL1,iL2,iL3,ndim]` (t first)
  (caller applies `ComplexF32.(...)` before writing)
