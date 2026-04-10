# src/ — Module Reference

Per-module API details live in each subfolder's own CLAUDE.md:

- [Dataset/CLAUDE.md](Dataset/CLAUDE.md) — HDF5 builders, merge, schemas
- [Preprocessing/CLAUDE.md](Preprocessing/CLAUDE.md) — split, normalization, data loading
- [Model/CLAUDE.md](Model/CLAUDE.md) — BaselineCNN, L-CNN layers, build_lcnn

## Flat modules (no subfolder)

### IO.jl

- `import_cern64(fname, ibc, lp; log=true)` → `Array{SU3{Float64},3}` shape `(bsz, ndim, rsz)`
  File direction order (t,x,y,z) remapped to LatticeGPU order (x,y,z,t) via `dtr = [4,1,2,3]`.
- `set_reader(fmt, lp)` — factory for format string `"cern"` (only supported format).

### Plaquette.jl

- `plaquette_field(U, lp)` → `Array{SU3{T},3}` shape `(bsz, npls, rsz)` — untraced P_μν(x).
- `plaquette_scalar_field(U, lp)` → `Array{Float64,3}` shape `(bsz, npls, rsz)` — Re(Tr P_μν(x)).

Plane ordering: `(4,1),(4,2),(4,3),(3,1),(3,2),(2,1)`.
Both dispatch only on `BC_PERIODIC`.

### Correlator.jl

- `LMAConfig` — mutable struct: `ncnfg`, `gamma`, `eigmodes`, `data::Dict`.
  `data["re"]` is `OrderedDict{String, Vector{Float64}}` keyed by source position.
- `read_contrib_all_sources(path, g)` — reads one `.dat` file; `tvals` detected dynamically.
- `get_LMAConfig_all_sources(path, g; em, bc, re_only)` — reads all sources from a config dir.
  File naming: `mseig{em}re.dat`. `em="VV"` → 64 modes, `em="PA"` → 32 modes.

## Known quirks

**`Re(Tr P)` normalization:** `LatticeGPU.tr(::SU3)` returns full complex trace.
For identity: `tr=3`. CLS average `Re(tr(P))≈0.5` is ~1/3 of CERN header `avgpl≈1.57`.

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
