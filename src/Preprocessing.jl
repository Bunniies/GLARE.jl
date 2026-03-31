module Preprocessing

using HDF5
using Statistics

export split_configs, NormStats, compute_normalization, save_normalization,
       load_normalization, load_config, load_split

# ---------------------------------------------------------------------------
# Train / val / test / bias_correction split
# ---------------------------------------------------------------------------

"""
    split_configs(h5path;
                  train=0.60, val=0.15, test=0.15, bias_corr=0.10)
        -> (train_ids, val_ids, test_ids, bias_corr_ids)

Partition the config ids in an HDF5 dataset into four non-overlapping splits
while **preserving Monte Carlo chain order** and **maximising separation**
between consecutive configs in the same split.

Config ids are sorted numerically (MC order). Assignment uses a generalised
Bresenham / DDA algorithm: iterating through the chain, each config is
assigned to whichever split is currently furthest below its target fraction.
This spreads each split uniformly throughout the chain rather than in
contiguous blocks, minimising autocorrelations within every split.

The `bias_corr` split contains configurations for which both the gauge field
and the expensive observable are available; they are used to apply an unbiased
correction to the NN prediction at inference time.

Returns four `Vector{String}` of config id keys in MC order.
"""
function split_configs(h5path::String;
                       train::Float64     = 0.60,
                       val::Float64       = 0.15,
                       test::Float64      = 0.15,
                       bias_corr::Float64 = 0.10)

    total = train + val + test + bias_corr
    abs(total - 1.0) < 1e-10 ||
        error("train + val + test + bias_corr must sum to 1.0, got $(total)")

    cfg_ids = h5open(h5path, "r") do fid
        sort(keys(fid["configs"]), by = x -> parse(Int, x))
    end

    fracs  = [train, val, test, bias_corr]
    labels = _interleaved_assign(length(cfg_ids), fracs)

    return (cfg_ids[labels .== 1],   # train
            cfg_ids[labels .== 2],   # val
            cfg_ids[labels .== 3],   # test
            cfg_ids[labels .== 4])   # bias_corr
end

# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------

"""
    NormStats

Holds per-channel mean and standard deviation for features and correlator,
computed from the training split only.

Fields:
- `feat_mean`  : `Vector{Float64}` length `npls` — mean of `plaq_scalar` per plane
- `feat_std`   : `Vector{Float64}` length `npls` — std  of `plaq_scalar` per plane
- `corr_mean`  : `Vector{Float64}` length `T`    — mean of correlator per time slice
- `corr_std`   : `Vector{Float64}` length `T`    — std  of correlator per time slice
"""
struct NormStats
    feat_mean :: Vector{Float64}
    feat_std  :: Vector{Float64}
    corr_mean :: Vector{Float64}
    corr_std  :: Vector{Float64}
end

"""
    compute_normalization(h5path, train_ids) -> NormStats

Compute per-plane feature statistics and per-time-slice correlator statistics
from the training split only. All sources are included when computing
correlator statistics.

Feature stats: mean and std of `plaq_scalar[b, ipl, r]` across all sites
(b, r) and all training configs, separately for each plane `ipl`.

Correlator stats: mean and std of `correlator[t, src]` across all sources
and all training configs, separately for each time slice `t`.
"""
function compute_normalization(h5path::String, train_ids::Vector{String})
    isempty(train_ids) && error("train_ids is empty")

    npls, T = h5open(h5path, "r") do fid
        grp = fid["configs"][train_ids[1]]
        size(read(grp["plaq_scalar"]), 2), size(read(grp["correlator"]), 1)
    end

    feat_sum  = zeros(Float64, npls)
    feat_sum2 = zeros(Float64, npls)
    feat_n    = zeros(Int64,   npls)

    corr_sum  = zeros(Float64, T)
    corr_sum2 = zeros(Float64, T)
    corr_n    = 0

    h5open(h5path, "r") do fid
        for cid in train_ids
            grp = fid["configs"][cid]

            ps = read(grp["plaq_scalar"])   # Float64[bsz, npls, rsz]
            for ipl in 1:npls
                vals = @view ps[:, ipl, :]
                feat_sum[ipl]  += sum(vals)
                feat_sum2[ipl] += sum(vals .^ 2)
                feat_n[ipl]    += length(vals)
            end

            co = read(grp["correlator"])    # Float64[T, nsrcs]
            for t in 1:T
                vals = @view co[t, :]
                corr_sum[t]  += sum(vals)
                corr_sum2[t] += sum(vals .^ 2)
            end
            corr_n += size(co, 2)
        end
    end

    feat_mean = feat_sum  ./ feat_n
    feat_var  = feat_sum2 ./ feat_n .- feat_mean .^ 2
    feat_std  = sqrt.(max.(feat_var, 0.0))

    n_corr    = corr_n * length(train_ids)
    corr_mean = corr_sum  ./ n_corr
    corr_var  = corr_sum2 ./ n_corr .- corr_mean .^ 2
    corr_std  = sqrt.(max.(corr_var, 0.0))

    feat_std[feat_std .< 1e-12] .= 1.0
    corr_std[corr_std .< 1e-12] .= 1.0

    return NormStats(feat_mean, feat_std, corr_mean, corr_std)
end

"""
    save_normalization(h5path, stats::NormStats)

Write normalization statistics into the `normalization/` group of an existing
HDF5 dataset file. Overwrites if the group already exists.
"""
function save_normalization(h5path::String, stats::NormStats)
    h5open(h5path, "r+") do fid
        haskey(fid, "normalization") && delete_object(fid, "normalization")
        grp = create_group(fid, "normalization")
        write(grp, "feat_mean", stats.feat_mean)
        write(grp, "feat_std",  stats.feat_std)
        write(grp, "corr_mean", stats.corr_mean)
        write(grp, "corr_std",  stats.corr_std)
    end
end

"""
    load_normalization(h5path) -> NormStats

Read normalization statistics previously written by `save_normalization`.
"""
function load_normalization(h5path::String)
    h5open(h5path, "r") do fid
        haskey(fid, "normalization") ||
            error("No normalization group found in $(h5path). " *
                  "Run compute_normalization + save_normalization first.")
        grp = fid["normalization"]
        NormStats(read(grp["feat_mean"]), read(grp["feat_std"]),
                  read(grp["corr_mean"]), read(grp["corr_std"]))
    end
end

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

"""
    load_config(h5path, cid; stats=nothing, field=:scalar)
        -> (features, correlator)

Load a single configuration from the HDF5 dataset.

- `features`   : `Float64[bsz, npls, rsz]` (scalar) or `ComplexF64[6, bsz, npls, rsz]` (matrix)
- `correlator` : `Float64[T, nsrcs]`

If `stats::NormStats` is provided, features and correlator are z-score normalized:
    `(x - mean) / std`
per plane for features and per time slice for the correlator.
Normalization is not applied to `plaq_matrix` (complex SU3 matrices).

The `field` keyword selects which plaquette representation to return:
- `:scalar`  — `plaq_scalar` (for baseline CNN)
- `:matrix`  — `plaq_matrix` (for equivariant L-CNN)
- `:both`    — returns a `NamedTuple (scalar=..., matrix=...)`
"""
function load_config(h5path::String, cid::String;
                     stats::Union{NormStats, Nothing} = nothing,
                     field::Symbol = :scalar)

    field in (:scalar, :matrix, :both) ||
        error("field must be :scalar, :matrix, or :both")

    features, correlator = h5open(h5path, "r") do fid
        haskey(fid["configs"], cid) ||
            error("Config \"$(cid)\" not found in $(h5path)")
        grp = fid["configs"][cid]
        if field in (:matrix, :both) && !haskey(grp, "plaq_matrix")
            error("plaq_matrix not found in config \"$(cid)\". " *
                  "Rebuild the dataset with save_matrix=true.")
        end
        feat = if field == :scalar
            read(grp["plaq_scalar"])
        elseif field == :matrix
            read(grp["plaq_matrix"])
        else
            (scalar = read(grp["plaq_scalar"]),
             matrix = read(grp["plaq_matrix"]))
        end
        feat, read(grp["correlator"])
    end

    stats === nothing && return features, correlator

    features   = _normalize_features(features, stats, field)
    correlator = _normalize_correlator(correlator, stats)
    return features, correlator
end

"""
    load_split(h5path, ids; stats=nothing, field=:scalar)
        -> (features_list, correlator_list)

Load all configurations in `ids` from the HDF5 dataset, in the order given.
Returns two `Vector`s. If `stats` is provided, data is normalized.
"""
function load_split(h5path::String, ids::Vector{String};
                    stats::Union{NormStats, Nothing} = nothing,
                    field::Symbol = :scalar)

    features_list   = Vector{Any}(undef, length(ids))
    correlator_list = Vector{Matrix{Float64}}(undef, length(ids))
    for (i, cid) in enumerate(ids)
        features_list[i], correlator_list[i] = load_config(h5path, cid;
                                                            stats=stats,
                                                            field=field)
    end
    return features_list, correlator_list
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _interleaved_assign(N, fracs) -> Vector{Int}

Assign each of N items to one of `length(fracs)` splits using a generalised
Bresenham / DDA algorithm. Iterating in order, each item is assigned to the
split whose accumulated deficit is largest. This spreads each split uniformly
throughout the sequence (maximally separated) without any shuffling.

`fracs` must sum to 1.0. Returns a label vector of integers in 1..nsplits.
"""
function _interleaved_assign(N::Int, fracs::Vector{Float64})
    nsplits = length(fracs)
    labels  = Vector{Int}(undef, N)
    acc     = zeros(Float64, nsplits)   # accumulated deficit per split

    for i in 1:N
        acc .+= fracs                   # each split "earns" its fraction
        j = argmax(acc)                 # most underserved split
        labels[i] = j
        acc[j] -= 1.0                   # "spend" one assignment
    end

    return labels
end

function _normalize_features(features, stats::NormStats, field::Symbol)
    if field == :scalar
        out = similar(features)
        for ipl in axes(features, 2)
            out[:, ipl, :] = (features[:, ipl, :] .- stats.feat_mean[ipl]) ./
                              stats.feat_std[ipl]
        end
        return out
    elseif field == :matrix
        return features   # complex matrices: not normalized via scalar stats
    else  # :both
        return (scalar = _normalize_features(features.scalar, stats, :scalar),
                matrix = features.matrix)
    end
end

function _normalize_correlator(correlator::Matrix{Float64}, stats::NormStats)
    out = similar(correlator)
    for t in axes(correlator, 1)
        out[t, :] = (correlator[t, :] .- stats.corr_mean[t]) ./ stats.corr_std[t]
    end
    return out
end

end # module Preprocessing
