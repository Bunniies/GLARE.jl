# ---------------------------------------------------------------------------
# Circular (periodic) padding — 3D spatial only (dims 1,2,3)
# ---------------------------------------------------------------------------

"""
    _circular_pad(x, pad)

Apply periodic boundary padding to a 5D array `x` of shape
`(d1, d2, d3, C, B)` by `pad[k]` sites on each side of spatial dimension k.
Uses `cat` to concatenate tail/head slices, which is AD-safe with Zygote.
"""
function _circular_pad(x::AbstractArray, pad::NTuple{3, Int})
    p1, p2, p3 = pad
    x = cat(x[end-p1+1:end, :, :, :, :], x, x[1:p1, :, :, :, :]; dims=1)
    x = cat(x[:, end-p2+1:end, :, :, :], x, x[:, 1:p2, :, :, :]; dims=2)
    x = cat(x[:, :, end-p3+1:end, :, :], x, x[:, :, 1:p3, :, :]; dims=3)
    return x
end

# ---------------------------------------------------------------------------
# PeriodicConv4D layer
# ---------------------------------------------------------------------------

"""
    PeriodicConv4D(ch_in => ch_out, kernel; activation=identity)

4D convolution layer with periodic (circular) boundary conditions,
implemented as a 3D spatial convolution with the temporal dimension
folded into the batch.

The temporal dimension (`Lt`, dim 1) is folded into the batch before
each forward pass and unfolded afterwards, so each time slice receives
the same spatial processing with shared weights. This is equivalent to
a true 4D conv for the baseline CNN where only spatial feature extraction
is needed prior to the spatial mean + MLP decoder.

Input shape:  `(Lt, Ls, Ls, Ls, ch_in,  B)`
Output shape: `(Lt, Ls, Ls, Ls, ch_out, B)` (same spatial size for odd kernels)

`kernel` is a 3-tuple specifying the 3D spatial kernel size, e.g. `(3,3,3)`.
"""
struct PeriodicConv4D
    conv :: Conv
    pad  :: NTuple{3, Int}
end

Flux.@layer PeriodicConv4D

function PeriodicConv4D(ch::Pair{Int, Int}, kernel::NTuple{3, Int};
                        activation = identity, kwargs...)
    pad = kernel .÷ 2
    PeriodicConv4D(Conv(kernel, ch, activation; pad = 0, kwargs...), pad)
end

function (l::PeriodicConv4D)(x::AbstractArray{T, 6}) where T
    Lt, Ls1, Ls2, Ls3, C_in, B = size(x)

    # (Lt, Ls, Ls, Ls, C_in, B) → (Ls, Ls, Ls, C_in, Lt, B)
    xp = permutedims(x, (2, 3, 4, 5, 1, 6))
    # fold Lt into batch → (Ls, Ls, Ls, C_in, Lt*B)
    xr = reshape(xp, Ls1, Ls2, Ls3, C_in, Lt * B)

    # 3D periodic conv
    yr = l.conv(_circular_pad(xr, l.pad))   # (Ls, Ls, Ls, C_out, Lt*B)

    C_out = size(yr, 4)
    # unfold Lt → (Ls, Ls, Ls, C_out, Lt, B)
    yp = reshape(yr, Ls1, Ls2, Ls3, C_out, Lt, B)
    # permute back → (Lt, Ls, Ls, Ls, C_out, B)
    return permutedims(yp, (5, 1, 2, 3, 4, 6))
end

# ---------------------------------------------------------------------------
# Baseline CNN factory
# ---------------------------------------------------------------------------

"""
    build_baseline_cnn(; Lt, npls, npol, channels, mlp_hidden) -> Chain

Construct the Phase-1 baseline CNN for correlator prediction.

Architecture:
```
PeriodicConv4D(npls  → channels[1], (3,3,3), relu)
PeriodicConv4D(ch[1] → channels[2], (3,3,3), relu)
...
spatial mean over dims 2,3,4 (Ls³)          → (Lt, ch_last, B)
Dense(Lt × ch_last → mlp_hidden, relu)
Dense(mlp_hidden   → Lt × npol)
reshape                                      → (Lt, npol, B)
```

Each `PeriodicConv4D` folds `Lt` into the batch dimension to apply a
3D spatial conv with shared weights across time slices, then unfolds.
Spatial periodicity is enforced by circular padding before each conv.

Input:  `(Lt, Ls, Ls, Ls, npls, B)` — normalised plaq_scalar, Float32
Output: `(Lt, npol, B)` — normalised correlator prediction (all polarizations)

Parameters:
- `Lt`:         temporal extent (default 48)
- `npls`:       number of plaquette planes, i.e. input channels (default 6)
- `npol`:       number of polarizations to predict simultaneously (default 3)
- `channels`:   hidden channel counts per conv layer (default [16, 16])
- `mlp_hidden`: hidden units in the MLP decoder (default 128)
"""
function build_baseline_cnn(;
        Lt         :: Int         = 48,
        npls       :: Int         = 6,
        npol       :: Int         = 3,
        channels   :: Vector{Int} = [16, 16],
        mlp_hidden :: Int         = 128)

    layers = Any[]
    in_ch = npls
    for ch_out in channels
        push!(layers, PeriodicConv4D(in_ch => ch_out, (3, 3, 3);
                                     activation = relu))
        in_ch = ch_out
    end

    final_ch = in_ch

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
