# src/Model/ — Model module

Neural network architectures for correlator prediction.
Split into two files plus the module entry point.

## File layout

```
Model.jl        — module entry, exports
BaselineCNN.jl  — Phase 1: PeriodicConv4D, build_baseline_cnn, pearson_r
LCNN.jl         — Phase 2: all L-CNN layers, LCBBlock, LCNN, build_lcnn
```

## Phase 1 — Baseline CNN (BaselineCNN.jl)

**`PeriodicConv4D(ch_in => ch_out, kernel; activation)`**
4D conv with exact circular padding. Folds `Lt` into batch for 3D NNlib conv.
Input/output: `(Lt, Ls, Ls, Ls, C, B)`. Kernel is a 3-tuple, e.g. `(3,3,3)`.
`_circular_pad` uses `cat`-based slicing — required for Zygote AD (fancy indexing gives zero grads).

**`build_baseline_cnn(; Lt=48, npls=6, npol=3, channels=[16,16], mlp_hidden=128)`** → `Flux.Chain`
`PeriodicConv4D(relu) × n → spatial mean (dims 2,3,4) → Dense(relu) → Dense → reshape`
Input: `(Lt,Ls,Ls,Ls,npls,B)`. Output: `(Lt,npol,B)`.

**`pearson_r(y_pred, y_true)`** → `Vector{Float64}` length Lt
Pearson r per time slice, pooled across all polarizations in the batch.

## Phase 2 — L-CNN (LCNN.jl)

### Internal helpers (AD-safe, `cat`/`reshape` only)

- `_roll(A, dim, shift)` — circular shift; `shift=-1` → forward neighbour, `shift=+1` → backward.
- `_pt_fwd(U_mu, W_j, dim)` — `U(x) · W(x+μ̂) · U†(x)` for all sites.
- `_pt_bwd(U_mu, W_j, dim)` — `U†(x-μ̂) · W(x-μ̂) · U(x-μ̂)` for all sites.
- `_su3_retrace(Φ)` — `Re(Tr Φ)`, `(rest...)` from `(3,3,rest...)`.

### Exported symbols

**`su3_reconstruct(x)`** — `(6,rest...) → (3,3,rest...)`. Third row via `conj(u1 × u2)`.

**`plaquette_matrices(U)`** — `(3,3,Lt,Ls,Ls,Ls,ndim,B) → (3,3,Lt,Ls,Ls,Ls,6,B)`.
Computes `P_μν(x) = U_μ(x)·U_ν(x+μ̂)·U†_μ(x+ν̂)·U†_ν(x)` for all 6 planes.
**Must be used as W₀** — raw links are NOT gauge-covariant (`V†` at different sites).
**Planes enumerated explicitly** (`_plane(4,1)..._plane(2,1)`) — no `push!`, AD-safe under Zygote.
Ordering `(4,1),(4,2),(4,3),(3,1),(3,2),(2,1)` matches `plaquette_field` (LatticeGPU convention: 1=x,2=y,3=z,4=t).
Direction→array-dim map: `(4,5,6,3)[mu]` (dim 3=t, dim 4=x, dim 5=y, dim 6=z). **NOT `mu+2`.**
Call site: `model(plaquette_matrices(U_batch), U_batch)` — plaquettes as W₀, links as U.

**`ScalarGate()`** — `σ(Re(Tr Φ)) ⊙ Φ` (sigmoid gate). No params. Gauge-covariant. Shape preserved. **Must use sigmoid, not relu** — relu causes forward-value explosion across stacked blocks since Re(Tr) is unbounded for non-SU(3) matrices.
Registered with `Functors.@functor` (NOT `Flux.@layer`) — `Flux.@layer` generates an `adapt_structure` that recurses infinitely on empty structs (no fields → Functors treats the struct itself as a leaf and re-enters). `Functors.@functor` sets `children = (;)` and reconstructs correctly.

**`TracePool()`** — `mean_x Re(Tr Φ(x))`. No params. Gauge-invariant.
`(3,3,Lt,Ls,Ls,Ls,C,B)` → `(Lt,C,B)`.
Same `Functors.@functor` registration as `ScalarGate` — same reason.

**`BilinearLayer(C_in1, C_in2, C_out)`** — `Φ_out[i] = Σ_{j,k} α[i,j,k] W_j W'_k`.
Learnable `α ∈ ℂ^{C_out×C_in1×C_in2}`. Gauge-covariant.
Use `(1,1,C_out,1) .* (3,3,1,N)` broadcasting — do NOT use `cat` across channel dim.

**`GaugeEquivConv(C_in, C_out; ndim=4)`** — L-Conv (arXiv:2012.12901 Eq. 5).
`Φ_out[i,x] = Σ_{j,μ} [ω[i,j,μ,1] PT_fwd + ω[i,j,μ,2] PT_bwd]`.
Learnable `ω ∈ ℝ^{C_out×C_in×ndim×2}`. Gauge-equivariant under site-dependent V(x).
Inputs: `W (3,3,...,C_in,B)`, `U (3,3,...,ndim,B)`. Output: `(3,3,...,C_out,B)`.
Direction→dim map: `(4,5,6,3)[mu]` (LatticeGPU 1=x,2=y,3=z,4=t → array dim 4,5,6,3). **NOT `mu+2`.**

**`LCBBlock(C_in, C_conv, C_out; ndim=4)`** — `GaugeEquivConv → BilinearLayer(W, W_conv) → ScalarGate`.
`BilinearLayer` takes `(W_local, W_transported)` — creates one-link Wilson loops per block;
each stacked block doubles the loop extent.

**`build_lcnn(; Lt=48, C_in=6, ndim=4, channels=[4,4], npol=3, mlp_hidden=64)`** → `LCNN`
`n` L-CB blocks → `TracePool` → MLP → `(Lt,npol,B)`. Same output signature as `build_baseline_cnn`.
MLP weights initialized in Float64 (change to Float32 for GPU training).

## Tensor layout convention

Matrix field Φ: `(3, 3, Lt, Ls, Ls, Ls, C, B)` — matrix indices first.
All `batched_mul`/trace ops fold spatial+batch into N: `reshape(permutedims(...), 3, 3, C, N)`.
