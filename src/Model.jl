module Model

using Flux
using NNlib
using Statistics

export PeriodicConv4D, build_baseline_cnn, pearson_r

# ---------------------------------------------------------------------------
# Circular (periodic) padding
# ---------------------------------------------------------------------------

"""
    _circular_pad(x, pad)

Apply periodic boundary padding to a 6D array `x` of shape
`(d1, d2, d3, d4, C, B)` by `pad[k]` sites on each side of dimension k.
Uses `mod1` indexing so the lattice wraps around exactly.
"""
function _circular_pad(x::AbstractArray, pad::NTuple{4, Int})
    d1, d2, d3, d4 = size(x, 1), size(x, 2), size(x, 3), size(x, 4)
    p1, p2, p3, p4 = pad
    idx1 = [mod1(i, d1) for i in (1 - p1):(d1 + p1)]
    idx2 = [mod1(i, d2) for i in (1 - p2):(d2 + p2)]
    idx3 = [mod1(i, d3) for i in (1 - p3):(d3 + p3)]
    idx4 = [mod1(i, d4) for i in (1 - p4):(d4 + p4)]
    return x[idx1, idx2, idx3, idx4, :, :]
end

# ---------------------------------------------------------------------------
# PeriodicConv4D layer
# ---------------------------------------------------------------------------

"""
    PeriodicConv4D(ch_in => ch_out, kernel; activation=identity)

4D convolution layer with periodic (circular) boundary conditions.

Wraps a `Flux.Conv` layer with zero explicit padding. Before each forward
pass the input is padded by `kernel .÷ 2` sites on each side using `mod1`
indexing, ensuring exact periodic wrap-around on all four lattice directions.

Input shape:  `(d1, d2, d3, d4, ch_in,  B)`
Output shape: `(d1, d2, d3, d4, ch_out, B)` (same spatial size for odd kernels)
"""
struct PeriodicConv4D
    conv :: Conv
    pad  :: NTuple{4, Int}
end

Flux.@layer PeriodicConv4D

function PeriodicConv4D(ch::Pair{Int, Int}, kernel::NTuple{4, Int};
                        activation = identity, kwargs...)
    pad = kernel .÷ 2
    PeriodicConv4D(Conv(kernel, ch, activation; pad = 0, kwargs...), pad)
end

function (l::PeriodicConv4D)(x::AbstractArray)
    l.conv(_circular_pad(x, l.pad))
end

# ---------------------------------------------------------------------------
# Baseline CNN factory
# ---------------------------------------------------------------------------

"""
    build_baseline_cnn(; Lt, Ls, npls, npol, channels, mlp_hidden) -> Chain

Construct the Phase-1 baseline CNN for correlator prediction.

Architecture:
```
PeriodicConv4D(npls  → channels[1], (3,3,3,3), relu)
PeriodicConv4D(ch[1] → channels[2], (3,3,3,3), relu)
...
spatial mean over dims 2,3,4 (Ls³)          → (Lt, ch_last, B)
Dense(Lt × ch_last → mlp_hidden, relu)
Dense(mlp_hidden   → Lt × npol)
reshape                                      → (Lt, npol, B)
```

Input:  `(Lt, Ls, Ls, Ls, npls, B)` — normalised plaq_scalar, Float32
Output: `(Lt, npol, B)` — normalised correlator prediction (all polarizations)

Parameters:
- `Lt`:         temporal extent (default 48)
- `Ls`:         spatial extent (default 24); only needed for shape checks
- `npls`:       number of plaquette planes, i.e. input channels (default 6)
- `npol`:       number of polarizations to predict simultaneously (default 3)
- `channels`:   hidden channel counts per conv layer (default [16, 16])
- `mlp_hidden`: hidden units in the MLP decoder (default 128)
"""
function build_baseline_cnn(;
        Lt         :: Int        = 48,
        npls       :: Int        = 6,
        npol       :: Int        = 3,
        channels   :: Vector{Int} = [16, 16],
        mlp_hidden :: Int        = 128)

    layers = Any[]
    in_ch = npls
    for ch_out in channels
        push!(layers, PeriodicConv4D(in_ch => ch_out, (3, 3, 3, 3);
                                     activation = relu))
        in_ch = ch_out
    end

    final_ch = in_ch

    # Spatial mean over dims 2, 3, 4 (each of size Ls); temporal is dim 1
    push!(layers,
          x -> dropdims(mean(x, dims = (2, 3, 4)), dims = (2, 3, 4)))  # (Lt, ch, B)

    push!(layers,
          x -> reshape(x, Lt * final_ch, :))                            # (Lt*ch, B)

    push!(layers, Dense(Lt * final_ch, mlp_hidden, relu))
    push!(layers, Dense(mlp_hidden, Lt * npol))

    push!(layers,
          x -> reshape(x, Lt, npol, :))                                 # (Lt, npol, B)

    return Chain(layers...)
end

# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

"""
    pearson_r(y_pred, y_true) -> Vector{Float64}

Pearson correlation coefficient per time slice, averaged over polarizations.

Arguments:
- `y_pred`, `y_true`: arrays of shape `(Lt, npol, B)` (normalised or raw)

Returns a `Vector{Float64}` of length `Lt` with r(t) pooled over all
polarizations (i.e. each polarization contributes B values).
"""
function pearson_r(y_pred::AbstractArray{<:Real, 3},
                   y_true::AbstractArray{<:Real, 3})
    Lt, npol, _ = size(y_pred)
    r = zeros(Float64, Lt)
    for t in 1:Lt
        preds = Float64[]
        trues = Float64[]
        for ipol in 1:npol
            append!(preds, vec(y_pred[t, ipol, :]))
            append!(trues, vec(y_true[t, ipol, :]))
        end
        r[t] = cor(preds, trues)
    end
    return r
end

end # module Model
