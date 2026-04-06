# src/ — Module Reference

## IO.jl

- `import_cern64(fname, ibc, lp; log=true)` → `Array{SU3{Float64},3}` shape `(bsz, ndim, rsz)`
  File direction order (t,x,y,z) remapped to LatticeGPU order (x,y,z,t) via `dtr = [4,1,2,3]`.
- `set_reader(fmt, lp)` — factory for format string `"cern"` (only supported format).

## Plaquette.jl

- `plaquette_field(U, lp)` → `Array{SU3{T},3}` shape `(bsz, npls, rsz)` — untraced P_μν(x).
  Formula: `U[b,id1,r] * U[bu1,id2,ru1] / (U[b,id2,r] * U[bu2,id1,ru2])`
  `/` = right-multiply by conjugate transpose (LatticeGPU convention).
- `plaquette_scalar_field(U, lp)` → `Array{Float64,3}` shape `(bsz, npls, rsz)` — Re(Tr P_μν(x)).

Plane ordering: `(4,1),(4,2),(4,3),(3,1),(3,2),(2,1)` — outer loop id1 descending, inner id2 ascending.
Both dispatch only on `BC_PERIODIC`.

## Correlator.jl

- `LMAConfig` — mutable struct: `ncnfg`, `gamma`, `eigmodes`, `data::Dict`.
  `data["re"]` is `OrderedDict{String, Vector{Float64}}` keyed by source position.
- `read_contrib_all_sources(path, g)` — reads one `.dat` file; `tvals` detected dynamically.
- `get_LMAConfig_all_sources(path, g; em, bc, re_only)` — reads all sources from a config dir.
  File naming: `mseig{em}re.dat`. `em="VV"` → 64 modes, `em="PA"` → 32 modes.

## Dataset.jl

### Functions

**`build_gauge_dataset(ensemble_path, lp, output_path; config_fmt, config_range, verbose)`**
Stores `plaq_scalar` in full spatial layout `Float64[iL[1],iL[2],iL[3],iL[4],npls]`.
`config_range::UnitRange{Int}` selects by position in sorted config list (e.g. `1:100`).

**`build_gauge_matrix_dataset(ensemble_path, lp, output_path; config_fmt, config_range, verbose)`**
Stores `plaq_matrix` in full spatial layout `ComplexF64[6,iL[1],iL[2],iL[3],iL[4],npls]`.
First 2 rows of SU3 only; third row implicit via unitarity. Separate file from scalar db.

**`build_corr_dataset(lma_path, output_path; em, polarizations, config_range, verbose)`**
Stores all three vector polarizations per config. Default `polarizations=["g1-g1","g2-g2","g3-g3"]`.

**`merge_dataset(input_paths, output_path; verbose)`**
Merge shard HDF5 files (produced with `config_range`) into one. Works for scalar, matrix, and
correlator databases. Metadata copied from first shard; `vol`/`svol` checked for consistency
across shards. Duplicate config ids raise an error.

### HDF5 schemas

```
gauge_scalar_db.h5                         gauge_matrix_db.h5
├── metadata/                              ├── metadata/
│   vol, svol, ensemble, config_fmt        │   vol, svol, ensemble, config_fmt
└── configs/<id>/                          └── configs/<id>/
    └── plaq_scalar                            └── plaq_matrix
        Float64[iL1,iL2,iL3,iL4,npls]             ComplexF64[6,iL1,iL2,iL3,iL4,npls]

corr_db.h5
├── metadata/  lma_path, em, polarizations
└── configs/<id>/<polarization>/
    ├── correlator  Float64[T, nsrcs]
    └── sources     String[nsrcs]
```

Config ids: trailing integer after `n` in gauge filenames (`A654r000n1` → `"1"`).
LMA dirs: plain integers. `config_range` indexes into the sorted list of available ids.

## Preprocessing.jl

- `split_configs(h5path; train, val, test, bias_corr)` → `(train_ids, val_ids, test_ids, bc_ids)`
  Interleaved (Bresenham) assignment — preserves MC chain order, maximises separation.
  Default: 60/15/15/10. `train_ids` ∩ `bc_ids` = ∅ by construction.
- `compute_normalization(gauge_h5, corr_h5, train_ids; polarizations)` → `NormStats`
  Per-plane mean/std for features (dim 5 of 5D array); per-t mean/std for correlator.
  Pooled across all polarizations. Never uses val/test/bc data.
- `save_normalization(h5path, stats)` / `load_normalization(h5path)`
- `load_gauge(gauge_h5, cid; stats, field)` → features
  `field=:scalar` → `Float64[iL1,iL2,iL3,iL4,npls]` (pass scalar db path)
  `field=:matrix` → `ComplexF64[6,iL1,iL2,iL3,iL4,npls]` (pass matrix db path)
  `field=:both` is **not supported** — load from the two separate databases individually.
- `load_corr(corr_h5, cid; stats, polarization)` → `Float64[T, nsrcs]`
- `load_config(gauge_h5, corr_h5, cid; stats, field, polarization)` → `(features, correlator)`
- `load_split(gauge_h5, corr_h5, ids; stats, field, polarization)` → `(feats_list, corrs_list)`

**Normalization:** global z-score on train only. Per-config normalization would destroy
the config-to-config fluctuations that are the regression signal.

## Model.jl

### Phase 1 — Baseline CNN

- `PeriodicConv4D(ch_in => ch_out, kernel; activation)` — 4D conv with exact circular padding.
  Wraps `Flux.Conv` with `pad=0`; pads input with `mod1` indexing before each forward pass.
  Input/output shape: `(Lt, Ls, Ls, Ls, C, B)`.
- `build_baseline_cnn(; Lt, npls, npol, channels, mlp_hidden)` → `Flux.Chain`
  Architecture: `PeriodicConv4D → relu → ... → spatial mean over dims(2,3,4) → MLP`
  Input: `(Lt, Ls, Ls, Ls, npls, B)`. Output: `(Lt, npol, B)`.
  Default: `channels=[16,16]`, `mlp_hidden=128`.
- `pearson_r(y_pred, y_true)` → `Vector{Float64}` length Lt
  Pearson r per time slice, pooled across all polarizations.

### Phase 2 — L-CNN

- `su3_reconstruct(x)` — recover full 3×3 SU(3) matrices from first-two-row storage.
  Input: `(6, rest...)` with entries `[u11,u12,u13,u21,u22,u23]` (layout from `build_gauge_matrix_dataset`).
  Output: `(3, 3, rest...)` — row index first, column second.
  Third row via `u3 = conj(u1 × u2)`. AD-safe (uses only `cat`/`reshape`).

## Known quirks

**`Re(Tr P)` normalization:** `LatticeGPU.tr(::SU3)` returns full complex trace.
For identity: `tr=3`. CLS average `Re(tr(P))≈0.5` is ~1/3 of CERN header `avgpl≈1.57`.
Values in `plaquette_scalar_field` are `Re(Tr P) ∈ [-3, 3]`.

**SU(3) closure test:** `dev_one(P)` = distance from identity, NOT SU(3)-ness.
Correct unitarity test: `dev_one(P / P) → 0`.

**SpaceParm field name:** block size is `lp.blk` (not `lp.bL`).

**CUDA circular dependency:** `CUDA` declares `__precompile__(false)`. Remove from
`Project.toml` until GPU support is actually implemented.

## LatticeGPU quick reference

| Symbol | Purpose |
|---|---|
| `SU3{T}` | SU(3) element — stores first 2 rows: `u11..u23` |
| `SpaceParm{N,M,B,D}` | Lattice geometry + block decomposition |
| `up((b,r), id, lp)` | Forward neighbour `(b,r)` in direction `id` |
| `dw((b,r), id, lp)` | Backward neighbour |
| `point_index(coord, lp)` | `CartesianIndex` → `(b, r)` |
| `point_coord((b,r), lp)` | `(b, r)` → `CartesianIndex` |
| `tr(::SU3)` | Complex trace: `u11+u22+conj(u11*u22-u12*u21)` |
| `dev_one(::SU3)` | Distance from identity |
| `BC_PERIODIC` | Boundary condition constant |

LatticeGPU installed at `~/.julia/packages/LatticeGPU/9VS4W/`.
