# GLARE — Claude Code Context

## Project goal

Build a gauge-equivariant neural network in Julia/Flux.jl that takes SU(3) gauge
configurations as input and predicts the rest-eigen correlator C(t) per configuration,
for variance reduction in lattice QCD ensembles.

Blueprint: arXiv:2602.21617 (supervised regression framework) + arXiv:2304.10438 (L-CNN
gauge-equivariant architecture). Full architecture plan is in PLAN.md.

## Repository layout

```
src/
  GLARE.jl        — top-level module, includes and re-exports submodules
  IO.jl           — module IO: gauge config readers (CERN, lex64, native, bsfqcd)
  Plaquette.jl    — module Plaquette: per-site plaquette fields (untraced + scalar)
  Correlator.jl   — module Correlator: LMA correlator reader (rest-eigen, ee, rr)
test/
  runtests.jl          — Pkg.test() entry point
  test_reader.jl       — IO tests: shape, unitarity of loaded links
  test_plaquette.jl    — Plaquette tests: shape, SU(3) closure, consistency, plane check
  test_correlator.jl   — Correlator tests: LMAConfig shape, source consistency, flags
PLAN.md           — full phased implementation plan
```

## Key dependencies

- **LatticeGPU** (`igit.ific.uv.es/alramos/latticegpu.jl`, Alberto Ramos group, IFIC Valencia)
  — installed at `~/.julia/packages/LatticeGPU/9VS4W/`
  — provides: `SU3{T}`, `SpaceParm`, `up`, `dw`, `point_index`, `point_coord`,
    `tr`, `dev_one`, `BC_PERIODIC` etc.
- **BDIO** — binary I/O
- **CUDA** — GPU support (not yet used in GLARE code; configs read to CPU)

## What is implemented

### `src/IO.jl`
- `import_cern64(fname, ibc, lp; log=true)` — reads OpenQCD/CERN double-precision
  binary format. Returns `Array{SU3{Float64}, 3}` of shape `(bsz, ndim, rsz)` (CPU).
  Direction convention in file: (t,x,y,z) → remapped to LatticeGPU order (x,y,z,t)
  via `dtr = [4,1,2,3]`.
- `set_reader(fmt, lp)` — factory returning the right reader closure for a given
  format string (`"cern"`, `"lex64"`, `"native"`, `"bsfqcd"`) and `SpaceParm`.

### `src/Plaquette.jl`
- `plaquette_field(U, lp)` — returns `Array{SU3{T},3}` of shape `(bsz, npls, rsz)`,
  containing the **untraced** Wilson plaquette `P_μν(x)` at every site and plane.
  Formula: `U[b,id1,r] * U[bu1,id2,ru1] / (U[b,id2,r] * U[bu2,id1,ru2])`
  where `/` = right-multiply by conjugate transpose (LatticeGPU convention).
- `plaquette_scalar_field(U, lp)` — returns `Array{Float64,3}` of shape `(bsz, npls, rsz)`,
  containing `Re(Tr P_μν(x))` at every site and plane — the 6-channel gauge-invariant
  scalar input for the baseline CNN.

Both functions currently dispatch only on `BC_PERIODIC`. Plane ordering: `(N,N-1), ..., (2,1)`
matching `lp.plidx`.

### `src/Correlator.jl`
- `LMAConfig` — mutable struct: `ncnfg::Int64`, `gamma::String`, `eigmodes::Int64`,
  `data::Dict{Any,Any}`. `data` has keys `"ee"`, `"re"`, `"rr"`, each an
  `OrderedDict{String, Vector{Float64}}` keyed by source position string.
- `read_contrib_all_sources(path, g)` — reads one `.dat` file (all sources in one file)
  for a given gamma structure. Returns `OrderedDict{String, Vector{Float64}}`.
  File format: `#tsrc=N` blocks, each containing `#<gamma>` sections with T rows of data.
  tvals is detected dynamically (no hardcoded lattice size).
- `get_LMAConfig_all_sources(path, g; em, bc, re_only)` — reads all three LMA contributions
  from a config directory. File naming: `mseig{em}ee.dat`, `mseig{em}re.dat`, `mseig{em}rr.dat`.
  `em="VV"` → 64 modes, `em="PA"` → 32 modes.
- Test config path: `ENV["GLARE_TEST_CORR"]` → directory for one config with all three `.dat` files.

## Known quirks

### `Re(Tr P)` normalization
`LatticeGPU.tr(::SU3)` returns the **full complex trace** `u11 + u22 + conj(u11*u22 - u12*u21)`.
For identity: `tr = 3`. For typical CLS plaquettes, `Re(tr(P))` ≈ 0.5 (per site/plane average),
which is ~1/3 of the value stored in the CERN file header (`avgpl ≈ 1.57`).
The relationship `avg_plaq * 3 ≈ avgpl` has been empirically confirmed but the exact
normalization difference between GLARE's computation and OpenQCD's header value is not yet
fully understood. The values in `plaquette_scalar_field` are `Re(Tr P) ∈ [-3, 3]` not
the normalized plaquette `Re(Tr P)/3`.

### SU(3) closure test
`dev_one(P)` measures distance from the **identity matrix**, NOT SU(3)-ness.
To test unitarity: use `dev_one(P / P)` which gives `dev_one(P · P†)` → 0 for SU(3).

### Config path
Default test config: `/Users/alessandroconigli/Lattice/data/cls/A654r000n1`
Override with `ENV["GLARE_TEST_CONF"]`. Tests skip gracefully if file is absent.
Ensemble: CLS A654, `VOL=(48,24,24,24)`, `SVOL=(8,4,4,4)`, `BC_PERIODIC`, no twist.

## What needs to be done next

### Phase 0 (remaining)
- [x] **Step 0.2a: Correlator reader** — `src/Correlator.jl`
  - `LMAConfig` struct, `read_contrib_all_sources`, `get_LMAConfig_all_sources`
  - Ported from `LmaPredict.jl/DataReader_all_t_sources.jl`
  - Test via `ENV["GLARE_TEST_CORR"]` pointing to a config dir with `mseig{em}ee/re/rr.dat`
- [ ] **Step 0.2b: Build dataset pipeline**
  - Loop over an ensemble of configs: load U, compute `plaquette_scalar_field` (baseline)
    or `plaquette_field` (equivariant), load C(t) via `get_LMAConfig_all_sources`, store as
    `(features, correlator)` pairs in HDF5 format.
  - Add HDF5.jl to deps; write `src/Dataset.jl` with `build_dataset` function.
- [ ] **Step 0.3: Data preprocessing**
  - Normalize features (zero mean, unit variance per plane-channel).
  - Normalize target C(t) (or use log-correlator).
  - Train/val/test split (70/15/15) — never mix configurations across splits.

### Phase 1 — Baseline CNN (gauge-invariant scalar inputs)
- [ ] Implement `PeriodicConv4D` — 4D convolution with periodic (circular) padding.
- [ ] Assemble baseline CNN: `PeriodicConv4D → relu → ... → spatial mean → MLP → C_pred(t)`.
- [ ] Write training loop with Adam optimiser + MSE loss.
- [ ] Evaluate: Pearson correlation r(t) and relative MSE per time slice.

### Phase 2 — Gauge-equivariant architecture (L-CNN)
- [ ] `GaugeEquivConv` layer: parallel transport + trainable scalar channel weights.
  Formula: `Σ_μ [w_μ U_μ(x) Φ(x+μ) U†_μ(x) + w_{-μ} U†_μ(x-μ) Φ(x-μ) U_μ(x-μ)]`
- [ ] `ScalarGate` nonlinearity: `σ(Re(Tr Φ)) * Φ` — preserves gauge equivariance.
- [ ] `BilinearLayer`: `Φ_out^a = Σ_{b,c} W_{abc} Φ^b · Φ^c`.
- [ ] `TraceAndAggregate`: `Re(Tr Φ(x))` → spatial mean per time slice → `[Lt, channels]`.
- [ ] Full model assembly + explicit gauge-equivariance unit test (transform U by random
  Ω(x) ∈ SU(3), assert `model(U^Ω) ≈ model(U)`).
- [ ] Add `ChainRulesCore.rrule` for SU3 matrix ops if Zygote AD fails.

### Phase 3 — Training and evaluation
- [ ] Weighted MSE loss (weight by `1/var(C(t))` across configs to handle late-time noise).
- [ ] Learning rate schedule (cosine annealing or ReduceLROnPlateau).
- [ ] Variance reduction factor: key metric from arXiv:2602.21617.
- [ ] Bias correction (control-variate estimator):
  `C_corrected(t) = C_pred(t) + mean_{labeled}(C_true - C_pred)(t)`

### Phase 4 — Extensions
- [ ] Larger Wilson loops (1×2 rectangles) as additional input channels.
- [ ] GPU support for `plaquette_field` / `plaquette_scalar_field` (CUDA kernels or
  CuArray-compatible broadcast).
- [ ] Gauge-equivariant pooling / multiscale U-Net hierarchy.
- [ ] Generalisation across β and L.

## LatticeGPU reference

Key types and functions used from LatticeGPU:
| Symbol | File | Purpose |
|---|---|---|
| `SU3{T}` | `Groups/SU3Types.jl` | SU(3) group element (stores first 2 rows) |
| `M3x3{T}` | `Groups/SU3Types.jl` | Full 3×3 complex matrix |
| `SpaceParm{N,M,B,D}` | `Space/Space.jl` | Lattice geometry + block decomposition |
| `up(p, id, lp)` | `Space/Space.jl` | Forward neighbour index |
| `dw(p, id, lp)` | `Space/Space.jl` | Backward neighbour index |
| `point_index(pt, lp)` | `Space/Space.jl` | CartesianIndex → (b,r) |
| `point_coord(p, lp)` | `Space/Space.jl` | (b,r) → CartesianIndex |
| `tr(::SU3)` | `Groups/GroupSU3.jl` | Complex trace: u11+u22+conj(u11*u22-u12*u21) |
| `dev_one(::SU3)` | `Groups/GroupSU3.jl` | Distance from identity (NOT SU3-ness) |
| `tensor_field(T, lp)` | `Fields/Fields.jl` | CuArray of shape (bsz, npls, rsz) |
| `BC_PERIODIC` etc. | `Space/Space.jl` | Boundary condition constants |
