# ---------------------------------------------------------------------------
# Internal helpers — AD-safe lattice operations
# ---------------------------------------------------------------------------

"""
    _roll(A, dim, shift)

Circularly shift array `A` by one position along `dim`.

`shift = -1`: `A[x] ← A[x+1]` (roll left, used to read the neighbour at x+μ̂).
`shift = +1`: `A[x] ← A[x-1]` (roll right, used to read the neighbour at x-μ̂).

Implemented via range-based `cat` slicing — AD-safe under Zygote (no fancy
integer-vector indexing).
"""
function _roll(A::AbstractArray, dim::Int, shift::Int)
    n  = size(A, dim)
    nd = ndims(A)
    if shift == -1
        head = A[ntuple(d -> d == dim ? (2:n) : Colon(), nd)...]
        tail = A[ntuple(d -> d == dim ? (1:1)  : Colon(), nd)...]
        return cat(head, tail; dims=dim)
    else   # shift == +1
        head = A[ntuple(d -> d == dim ? (n:n)   : Colon(), nd)...]
        tail = A[ntuple(d -> d == dim ? (1:n-1) : Colon(), nd)...]
        return cat(head, tail; dims=dim)
    end
end

# Forward parallel transport at every site simultaneously:
#   PT_fwd(U_μ, W_j)[x] = U_μ(x) · W_j(x+μ̂) · U_μ†(x)
#
# U_μ, W_j: (3, 3, rest...) — single-channel matrix fields.
# dim: the array dimension corresponding to direction μ (e.g. dim=3 for t, dim=4 for x).
function _pt_fwd(U_mu::AbstractArray, W_j::AbstractArray, dim::Int)
    W_sh  = _roll(W_j, dim, -1)          # W_j(x+mu) gathered at x
    N     = prod(size(W_j)[3:end])
    Uf    = reshape(U_mu,  3, 3, N)
    Wf    = reshape(W_sh,  3, 3, N)
    Uf_adj = conj.(permutedims(Uf, (2, 1, 3)))
    return reshape(NNlib.batched_mul(NNlib.batched_mul(Uf, Wf), Uf_adj), size(W_j))
end

# Backward parallel transport:
#   PT_bwd(U_mu, W_j)[x] = U_mu_adj(x-mu) * W_j(x-mu) * U_mu(x-mu)
function _pt_bwd(U_mu::AbstractArray, W_j::AbstractArray, dim::Int)
    W_sh   = _roll(W_j,  dim, +1)          # W_j(x-mu) gathered at x
    U_sh   = _roll(U_mu, dim, +1)          # U_mu(x-mu) gathered at x
    N      = prod(size(W_j)[3:end])
    Uf     = reshape(U_sh,  3, 3, N)
    Wf     = reshape(W_sh,  3, 3, N)
    Uf_adj = conj.(permutedims(Uf, (2, 1, 3)))
    return reshape(NNlib.batched_mul(NNlib.batched_mul(Uf_adj, Wf), Uf), size(W_j))
end

# ---------------------------------------------------------------------------
# Plaquette matrix field (gauge-covariant W₀ for L-CNN input)
# ---------------------------------------------------------------------------

"""
    plaquette_matrices(U) -> Array{Complex{T}, 8}

Compute the gauge-covariant plaquette matrix field from gauge links.

```
P_μν(x) = U_μ(x) · U_ν(x+μ̂) · U†_μ(x+ν̂) · U†_ν(x)
```

Each plaquette matrix transforms as `P_μν(x) → V(x) P_μν(x) V†(x)` under
site-dependent gauge transformations, making it a valid gauge-covariant
initial field `W₀` for the L-CNN (unlike raw links, which transform with
different `V` at each endpoint).

Input:  `U` of shape `(3, 3, Lt, Ls, Ls, Ls, ndim, B)`
Output: `P` of shape `(3, 3, Lt, Ls, Ls, Ls, npls, B)` where `npls = ndim*(ndim-1)/2`

Plane ordering: `(4,1),(4,2),(4,3),(3,1),(3,2),(2,1)` — 6 planes for a 4D lattice
using LatticeGPU direction convention (1=x, 2=y, 3=z, 4=t), matching `Plaquette.jl`.
"""
function plaquette_matrices(U::AbstractArray{<:Complex, 8})
    sz   = size(U)   # (3, 3, Lt, Ls, Ls, Ls, ndim, B)
    Lt, Ls, _, B = sz[3], sz[4], sz[7], sz[8]
    N    = Lt * Ls^3 * B

    _adj(A) = conj.(permutedims(A, (2, 1, 3, 4, 5, 6, 7)))
    _mul(A, C) = reshape(NNlib.batched_mul(reshape(A, 3, 3, N), reshape(C, 3, 3, N)),
                         3, 3, Lt, Ls, Ls, Ls, B)

    # Map LatticeGPU direction mu (1=x, 2=y, 3=z, 4=t) to the array dimension in the
    # 7D single-channel field (3,3,Lt,Ls,Ls,Ls,B): dim 3=t, dim 4=x, dim 5=y, dim 6=z.
    _ddir(mu) = (4, 5, 6, 3)[mu]

    # P_μν(x) = U_μ(x) · U_ν(x+μ̂) · U†_μ(x+ν̂) · U†_ν(x)
    function _plane(mu, nu)
        U_mu = U[:, :, :, :, :, :, mu, :]
        U_nu = U[:, :, :, :, :, :, nu, :]
        reshape(_mul(U_mu,
                 _mul(_roll(U_nu, _ddir(mu), -1),
                  _mul(_adj(_roll(U_mu, _ddir(nu), -1)),
                       _adj(U_nu)))),
                3, 3, Lt, Ls, Ls, Ls, 1, B)
    end

    # Enumerate all 6 planes explicitly — no push!/mutation, AD-safe under Zygote.
    # Plane ordering: (4,1),(4,2),(4,3),(3,1),(3,2),(2,1) — matches Plaquette.jl:
    # (t,x),(t,y),(t,z),(z,x),(z,y),(y,x) in LatticeGPU direction convention.
    return cat(_plane(4,1), _plane(4,2), _plane(4,3),
               _plane(3,1), _plane(3,2), _plane(2,1); dims=7)
end

# ---------------------------------------------------------------------------
# SU(3) utilities
# ---------------------------------------------------------------------------

"""
    su3_reconstruct(x) -> Array{Complex{T}, N}

Reconstruct full 3×3 SU(3) matrices from their first-two-row storage format.

Input `x` has shape `(6, rest...)` where dim 1 stores the 6 complex entries
`[u11, u12, u13, u21, u22, u23]` of each SU(3) matrix (the first two rows).
This is the layout produced by `build_gauge_link_dataset`.

The third row is recovered from the SU(3) unitarity constraint via the
complex conjugate of the cross product of rows 1 and 2:

```
u31 = conj(u12*u23 - u13*u22)
u32 = conj(u13*u21 - u11*u23)
u33 = conj(u11*u22 - u12*u21)
```

Output shape is `(3, 3, rest...)` with dim 1 indexing rows and dim 2 columns.

Uses only `cat` and `reshape` — AD-safe under Zygote.
"""
function su3_reconstruct(x::AbstractArray{<:Complex})
    rest = size(x)[2:end]
    nd   = ndims(x)

    _s(i) = x[i:i, ntuple(_ -> Colon(), nd - 1)...]

    u11, u12, u13 = _s(1), _s(2), _s(3)
    u21, u22, u23 = _s(4), _s(5), _s(6)

    u31 = conj.(u12 .* u23 .- u13 .* u22)
    u32 = conj.(u13 .* u21 .- u11 .* u23)
    u33 = conj.(u11 .* u22 .- u12 .* u21)

    row1 = reshape(cat(u11, u12, u13; dims=1), 1, 3, rest...)
    row2 = reshape(cat(u21, u22, u23; dims=1), 1, 3, rest...)
    row3 = reshape(cat(u31, u32, u33; dims=1), 1, 3, rest...)

    return cat(row1, row2, row3; dims=1)
end

# Internal helper: Re(Tr Φ) for Φ of shape (3, 3, rest...).
# Uses scalar indexing on dims 1,2 only — AD-safe under Zygote.
function _su3_retrace(Φ::AbstractArray{<:Complex})
    nd   = ndims(Φ)
    rest = ntuple(_ -> Colon(), nd - 2)
    return real.(Φ[1, 1, rest...] .+ Φ[2, 2, rest...] .+ Φ[3, 3, rest...])
end

# ---------------------------------------------------------------------------
# BilinearLayer
# ---------------------------------------------------------------------------

"""
    BilinearLayer(C_in1, C_in2, C_out)

Gauge-covariant bilinear layer (L-Bilin, arXiv:2012.12901 Eq. 6).

For each output channel `i` and each lattice site, computes:

```
Φ_out[:,:,i,...] = Σ_{j,k} α[i,j,k] * W[:,:,j,...] * W'[:,:,k,...]
```

where `*` is 3×3 matrix multiplication and `α ∈ ℂ^{C_out × C_in1 × C_in2}`
are the learnable complex weights.

Takes two matrix field tensors as input (typically the local field W and the
parallel-transported field W' produced by `GaugeEquivConv`).

Input:  `W`  of shape `(3, 3, Lt, Ls, Ls, Ls, C_in1, B)`
        `W'` of shape `(3, 3, Lt, Ls, Ls, Ls, C_in2, B)`
Output: shape `(3, 3, Lt, Ls, Ls, Ls, C_out, B)`

Gauge-covariant: if `W → V(x) W V†(x)` and `W' → V(x) W' V†(x)` then
`Φ_out → V(x) Φ_out V†(x)`, since `(VWV†)(VW'V†) = V(WW')V†`.
"""
struct BilinearLayer{A}
    α :: A   # (C_out, C_in1, C_in2)
end

Flux.@layer BilinearLayer

function BilinearLayer(C_in1::Int, C_in2::Int, C_out::Int)
    BilinearLayer(randn(ComplexF32, C_out, C_in1, C_in2) ./ sqrt(Float32(C_in1 * C_in2)))
end

function (l::BilinearLayer)(W::AbstractArray{<:Complex, 8},
                             W′::AbstractArray{<:Complex, 8})
    sz    = size(W)
    C_in1 = size(l.α, 2)
    C_in2 = size(l.α, 3)
    C_out = size(l.α, 1)
    N     = sz[3] * sz[4] * sz[5] * sz[6] * sz[8]   # Lt*Ls³*B

    # Move channel to dim 3, fold spatial+batch → N: (3, 3, C, N)
    W_flat  = reshape(permutedims(W,  (1, 2, 7, 3, 4, 5, 6, 8)), 3, 3, C_in1, N)
    W′_flat = reshape(permutedims(W′, (1, 2, 7, 3, 4, 5, 6, 8)), 3, 3, C_in2, N)

    # Two-step batched contraction — O(1) Zygote nodes instead of C_in1*C_in2.
    #
    # Step 1: contract α with W' to build V[c,b,i,j,n] = Σ_k α[i,j,k] * W'[c,b,k,n]
    #   α:      (C_out, C_in1, C_in2) → reshape (C_out*C_in1, C_in2)   [i varies fastest]
    #   W'_flat: (3, 3, C_in2, N) → permute(3,1,2,4) (C_in2,3,3,N) → reshape (C_in2, 9*N)
    #   product: (C_out*C_in1, 9*N) → reshape (C_out, C_in1, 3, 3, N) → permute (3,4,1,2,5)
    #   V:       (3, 3, C_out, C_in1, N)
    α_mat  = reshape(l.α, C_out * C_in1, C_in2)
    W′_mat = reshape(permutedims(W′_flat, (3, 1, 2, 4)), C_in2, 9 * N)
    V      = permutedims(reshape(α_mat * W′_mat, C_out, C_in1, 3, 3, N), (3, 4, 1, 2, 5))
    # V: (3, 3, C_out, C_in1, N)  i.e. (c, b, i, j, n)

    # Step 2: contract W with V via batched_mul
    #   out[a,b,i,n] = Σ_{c,j} W[a,c,j,n] * V[c,b,i,j,n]
    #   Fold (c,j) into one index p (c varies fastest):
    #   W_left:  (3, 3*C_in1, N)        — reshape(W_flat, 3, 3*C_in1, N)
    #   V_right: (3*C_in1, 3*C_out, N)  — permute V(c,b,i,j,n)→(c,j,b,i,n) → reshape
    #   batched_mul → (3, 3*C_out, N) → reshape (3, 3, C_out, N)
    W_left  = reshape(W_flat, 3, 3 * C_in1, N)
    V_right = reshape(permutedims(V, (1, 4, 2, 3, 5)), 3 * C_in1, 3 * C_out, N)
    out_4d  = reshape(NNlib.batched_mul(W_left, V_right), 3, 3, C_out, N)

    # Restore layout: (3, 3, C_out, N) → (3, 3, C_out, Lt, Ls, Ls, Ls, B) → (3, 3, Lt, Ls, Ls, Ls, C_out, B)
    out_8d = reshape(out_4d, 3, 3, C_out, sz[3], sz[4], sz[5], sz[6], sz[8])
    return permutedims(out_8d, (1, 2, 4, 5, 6, 7, 3, 8))
end

# ---------------------------------------------------------------------------
# ScalarGate
# ---------------------------------------------------------------------------

"""
    ScalarGate()

Pointwise nonlinearity for gauge-covariant matrix fields:

```
ScalarGate(Φ) = σ(Re(Tr Φ)) ⊙ Φ
```

where σ = sigmoid is applied elementwise to the real trace (the L-Act layer of
arXiv:2012.12901, Eq. 7), bounding the gate to (0,1) to prevent forward-value
explosion across stacked blocks. The result broadcasts over the 3×3 matrix indices.

Input/output shape: `(3, 3, Lt, Ls, Ls, Ls, C, B)`.

The operation is gauge-covariant: if `Φ(x) → V(x) Φ(x) V†(x)` then
`Re(Tr Φ)` is gauge-invariant (the scalar gate value is unchanged) and
the output transforms identically to the input.

No learnable parameters.
"""
struct ScalarGate end

Functors.@functor ScalarGate

function (::ScalarGate)(Φ::AbstractArray{<:Complex})
    # Φ: (3, 3, Lt, Ls, Ls, Ls, C, B)
    # Re(Tr Φ): (Lt, Ls, Ls, Ls, C, B)
    gate = sigmoid.(_su3_retrace(Φ)) # sigmoid
    # gate = relu.(_su3_retrace(Φ)) # relu
    # reshape to (1, 1, Lt, Ls, Ls, Ls, C, B) for broadcasting over matrix dims
    return reshape(gate, 1, 1, size(gate)...) .* Φ
end

# ---------------------------------------------------------------------------
# TracePool
# ---------------------------------------------------------------------------

"""
    TracePool()

Readout layer that maps a gauge-covariant matrix field to a real scalar field
via trace projection followed by spatial mean pooling:

```
TracePool(Φ) = mean_{x∈Ls³} Re(Tr Φ(x))
```

Input shape:  `(3, 3, Lt, Ls, Ls, Ls, C, B)`
Output shape: `(Lt, C, B)`

The spatial mean is taken over the three spatial dimensions (dims 4,5,6 of
the input, which become dims 2,3,4 of the real trace). The temporal dimension
and channel dimension are preserved, matching the shape expected by the
MLP decoder.

No learnable parameters.
"""
struct TracePool end

Functors.@functor TracePool

function (::TracePool)(Φ::AbstractArray{<:Complex})
    # Φ: (3, 3, Lt, Ls, Ls, Ls, C, B)
    # Re(Tr Φ): (Lt, Ls, Ls, Ls, C, B)
    tr_Φ = _su3_retrace(Φ)
    # spatial mean over Ls dims (2, 3, 4) → (Lt, C, B)
    return dropdims(mean(tr_Φ, dims=(2, 3, 4)), dims=(2, 3, 4))
end

# ---------------------------------------------------------------------------
# GaugeEquivConv  (L-Conv, arXiv:2012.12901 Eq. 5)
# ---------------------------------------------------------------------------

"""
    GaugeEquivConv(C_in, C_out; ndim=4)

Gauge-equivariant convolution layer (L-Conv) implementing parallel transport
along all `ndim` lattice directions (arXiv:2012.12901, Eq. 5):

```
Φ_out[i,x] = Σ_{j,μ} [ ω[i,j,μ,1] PT_fwd(U_μ,W_j)[x]
                       + ω[i,j,μ,2] PT_bwd(U_μ,W_j)[x] ]
```

where the two parallel transports are:

```
PT_fwd(U_μ, W_j)[x] = U_μ(x)    · W_j(x+μ̂) · U_μ(x)†
PT_bwd(U_μ, W_j)[x] = U_μ(x-μ̂)† · W_j(x-μ̂) · U_μ(x-μ̂)
```

`ω ∈ ℝ^{C_out × C_in × ndim × 2}` are the learnable real weights.

Inputs:
- `W`: matter field  `(3, 3, Lt, Ls, Ls, Ls, C_in, B)` — gauge-covariant
- `U`: gauge links   `(3, 3, Lt, Ls, Ls, Ls, ndim, B)` — full SU(3) matrices

Output: `(3, 3, Lt, Ls, Ls, Ls, C_out, B)` — gauge-covariant.

**Gauge equivariance:** under the site-dependent transformation
`W(x) → V(x) W(x) V†(x)` and `U_μ(x) → V(x) U_μ(x) V†(x+μ̂)`,
the output satisfies `Φ_out(x) → V(x) Φ_out(x) V†(x)`.

Direction mapping: μ=1 → dim 4 (Ls), μ=2 → dim 5 (Ls), μ=3 → dim 6 (Ls), μ=4 → dim 3 (Lt).

Boundary conditions are periodic (implemented via `_roll`).
"""
struct GaugeEquivConv{A}
    omega :: A   # (C_out, C_in, ndim, 2)
end

Flux.@layer GaugeEquivConv

function GaugeEquivConv(C_in::Int, C_out::Int; ndim::Int=4)
    GaugeEquivConv(randn(Float32, C_out, C_in, ndim, 2) ./ sqrt(Float32(C_in * 2 * ndim)))
end

function (l::GaugeEquivConv)(W::AbstractArray{<:Complex, 8},
                              U::AbstractArray{<:Complex, 8})
    # W: (3, 3, Lt, Ls, Ls, Ls, C_in, B)
    # U: (3, 3, Lt, Ls, Ls, Ls, ndim, B)
    sz    = size(W)
    C_in  = size(l.omega, 2)
    C_out = size(l.omega, 1)
    ndim  = size(l.omega, 3)
    N     = sz[3] * sz[4] * sz[5] * sz[6] * sz[8]   # Lt*Ls³*B

    # Build PT_all using Zygote.Buffer — O(C_in*ndim*N) memory, no quadratic
    # cat chain. Each transport is written to a pre-sized buffer slot; copy(buf)
    # produces the final array. Zygote records each slice assignment and
    # distributes gradients back to the individual transports.
    n_ch = C_in * ndim * 2
    buf  = Zygote.Buffer(W, 3, 3, n_ch, N)
    for j in 1:C_in, mu in 1:ndim
        mu_dim = (4, 5, 6, 3)[mu]
        U_mu   = U[:, :, :, :, :, :, mu, :]
        W_j    = W[:, :, :, :, :, :, j,  :]
        p      = (j - 1) * ndim + mu   # pair index, 1-based
        buf[:, :, 2p-1, :] = reshape(_pt_fwd(U_mu, W_j, mu_dim), 3, 3, N)
        buf[:, :, 2p,   :] = reshape(_pt_bwd(U_mu, W_j, mu_dim), 3, 3, N)
    end
    PT_all = copy(buf)   # (3, 3, C_in*ndim*2, N)

    # Contract with omega via a single matrix multiply.
    # Channel ordering: p(j,mu) with channels 2p-1(fwd), 2p(bwd).
    # 0-indexed: (dir-1) + 2*(mu-1) + 2*ndim*(j-1)  [dir fastest, mu next, j slowest].
    # omega permuted to (C_out, dir, mu, j) so reshape matches this ordering.
    omega_mat = reshape(permutedims(l.omega, (1, 4, 3, 2)), C_out, n_ch)
    # PT_all: (3,3,C_in*ndim*2,N) → permute(3,1,2,4) → (C_in*ndim*2, 3, 3, N) → reshape (C_in*ndim*2, 9*N)
    PT_mat    = reshape(permutedims(PT_all, (3, 1, 2, 4)), C_in * ndim * 2, 9 * N)
    # (C_out, C_in*ndim*2) × (C_in*ndim*2, 9*N) → (C_out, 9*N) → (C_out,3,3,N) → (3,3,C_out,N)
    out_4d    = permutedims(reshape(omega_mat * PT_mat, C_out, 3, 3, N), (2, 3, 1, 4))

    # (3, 3, C_out, N) → (3, 3, C_out, Lt, Ls, Ls, Ls, B) → (3, 3, Lt, Ls, Ls, Ls, C_out, B)
    out_8d = reshape(out_4d, 3, 3, C_out, sz[3], sz[4], sz[5], sz[6], sz[8])
    return permutedims(out_8d, (1, 2, 4, 5, 6, 7, 3, 8))
end

# ---------------------------------------------------------------------------
# LCBBlock  (L-Conv–Bilinear block, arXiv:2012.12901 §III)
# ---------------------------------------------------------------------------

"""
    LCBBlock(C_in, C_conv, C_out; ndim=4)

Lattice Convolution-Bilinear block — the core repeated unit of the L-CNN.

Each block performs three gauge-covariant operations:

```
W_conv = GaugeEquivConv(W, U)          # parallel transport: C_in → C_conv
W_out  = BilinearLayer(W, W_conv)      # bilinear product:  (C_in, C_conv) → C_out
W_out  = ScalarGate(W_out)             # nonlinearity:      ReLU(Re(Tr)) ⊙ Φ
```

The bilinear layer combines the *local* field `W` with the *transported* field
`W_conv`, creating matrix products `W(x) · U(x)W(x+μ̂)U†(x)` that trace out
Wilson loops of increasing size. Each stacked L-CB block doubles the effective
loop extent: 1 block → 1-link paths, 2 blocks → 2-link paths, etc.

Inputs:
- `W`: matrix field  `(3, 3, Lt, Ls, Ls, Ls, C_in, B)` — gauge-covariant
- `U`: gauge links   `(3, 3, Lt, Ls, Ls, Ls, ndim, B)` — full SU(3) matrices

Output: `(3, 3, Lt, Ls, Ls, Ls, C_out, B)` — gauge-covariant.

Gauge links `U` are passed through unchanged (they are fixed background fields,
not updated by the block).
"""
struct LCBBlock
    conv :: GaugeEquivConv
    bilin :: BilinearLayer
    gate :: ScalarGate
end

Flux.@layer LCBBlock

function LCBBlock(C_in::Int, C_conv::Int, C_out::Int; ndim::Int=4)
    LCBBlock(
        GaugeEquivConv(C_in, C_conv; ndim=ndim),
        BilinearLayer(C_in, C_conv, C_out),
        ScalarGate()
    )
end

function (l::LCBBlock)(W::AbstractArray{<:Complex, 8},
                        U::AbstractArray{<:Complex, 8})
    # No sub-layer checkpoint: the block-level checkpoint in LCNN already reruns the
    # entire block during backward, so an inner checkpoint would cause conv to run 4x
    # (2x from block rerun × 2x from inner rerun). Without it, conv runs 2x total.
    W_conv = l.conv(W, U)
    W_out  = l.bilin(W, W_conv)
    return l.gate(W_out)
end

# ---------------------------------------------------------------------------
# LCNN  (full L-CNN model)
# ---------------------------------------------------------------------------

"""
    LCNN(blocks, pool, mlp, Lt, npol)

Full lattice gauge-equivariant CNN: stacked L-CB blocks → TracePool → MLP decoder.

Data flow:
```
gauge links (6, iL1, ..., iL4, ndim)  ──→  su3_reconstruct  ──→  U (3,3,Lt,Ls³,ndim,B)
                                                                     │
input plaquette matrices (6, iL1, ..., iL4, npls) → su3_reconstruct → W₀ (3,3,Lt,Ls³,C_in,B)
                                                                     │
    W₁ = LCBBlock₁(W₀, U)                                          │ gauge-covariant
    W₂ = LCBBlock₂(W₁, U)                                          │
    ...                                                              ↓
    scalar = TracePool(W_n)                                 (Lt, C_last, B)  gauge-invariant
    flatten → Dense → relu → Dense → reshape                (Lt, npol, B)
```

Use [`build_lcnn`](@ref) to construct.

The forward pass takes **two** arguments `(W, U)`:
- `W`: initial matrix field `(3, 3, Lt, Ls, Ls, Ls, C_in, B)` (e.g. plaquette matrices)
- `U`: gauge links `(3, 3, Lt, Ls, Ls, Ls, ndim, B)` (full SU(3), from `su3_reconstruct`)

Output: `(Lt, npol, B)` — predicted correlator (normalised), same format as the baseline CNN.
"""
struct LCNN
    blocks :: Vector{LCBBlock}
    pool   :: TracePool
    mlp    :: Chain
    Lt     :: Int
    npol   :: Int
end

Flux.@layer LCNN

function (m::LCNN)(W::AbstractArray{<:Complex, 8},
                    U::AbstractArray{<:Complex, 8})
    for blk in m.blocks
        W = Zygote.checkpointed(blk, W, U)
    end
    x = m.pool(W)                                   # (Lt, C_last, B)  real
    x = reshape(x, m.Lt * size(x, 2), :)            # (Lt*C_last, B)
    x = m.mlp(x)                                    # (Lt*npol, B)
    return reshape(x, m.Lt, m.npol, :)              # (Lt, npol, B)
end

"""
    build_lcnn(; Lt, C_in, ndim, channels, npol, mlp_hidden) -> LCNN

Construct a full L-CNN model for correlator prediction from gauge fields.

Architecture: `n` stacked L-CB blocks → TracePool → MLP decoder.

Each L-CB block `k` has channel widths `(channels[k], channels[k+1], channels[k+1])`:
the internal convolution channel `C_conv` equals `C_out` for simplicity; all three
widths are independently settable via the `LCBBlock` constructor if needed.

Parameters:
- `Lt`:         temporal extent (default 48)
- `C_in`:       input channels — number of plaquette planes for matrix input (default 6)
- `ndim`:       number of lattice directions (default 4)
- `channels`:   hidden channel widths per L-CB block (default `[4, 4]`)
- `npol`:       number of polarizations to predict simultaneously (default 3)
- `mlp_hidden`: hidden units in the MLP decoder (default 64)
"""
function build_lcnn(;
        Lt         :: Int         = 48,
        C_in       :: Int         = 6,
        ndim       :: Int         = 4,
        channels   :: Vector{Int} = [4, 4],
        npol       :: Int         = 3,
        mlp_hidden :: Int         = 64)

    blocks = LCBBlock[]
    ch_in = C_in
    for ch_out in channels
        push!(blocks, LCBBlock(ch_in, ch_out, ch_out; ndim=ndim))
        ch_in = ch_out
    end

    C_last = channels[end]
    pool   = TracePool()
    mlp  = Chain(
        Dense(Lt * C_last => mlp_hidden, relu; init=Flux.glorot_uniform),
        Dense(mlp_hidden => Lt * npol; init=Flux.glorot_uniform)
    )

    return LCNN(blocks, pool, mlp, Lt, npol)
end
