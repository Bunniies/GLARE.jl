module Preprocessing

using HDF5
using Statistics

export split_configs, NormStats, compute_normalization, save_normalization,
       load_normalization, load_gauge, load_corr, load_config, load_split,
       PreloadedDataset, preload_dataset

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

The `bias_corr` split contains configurations used to apply an unbiased
correction to the NN prediction at inference time. It must never overlap
with `train_ids`. Pass either the gauge or the correlator HDF5 path —
both share the same `configs/<id>` key structure.

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
    compute_normalization(gauge_h5, corr_h5, train_ids;
                          polarizations = ["g1-g1", "g2-g2", "g3-g3"]) -> NormStats

Compute per-plane feature statistics and per-time-slice correlator statistics
from the training split only. Feature stats come from `plaq_scalar` in
`gauge_h5`; correlator stats are pooled across all `polarizations` in `corr_h5`.

Feature stats: mean and std of `plaq_scalar[b, ipl, r]` across all sites
(b, r) and all training configs, separately for each plane `ipl`.

Correlator stats: mean and std of the **source-averaged** correlator `C̄(t)`
across all training configs and all polarizations, separately for each time
slice `t`. Stats are computed on source-averaged data to match the training
target, so that the normalized targets have unit variance.

**Never uses val/test/bc data.**
"""
function compute_normalization(gauge_h5::String, corr_h5::String,
                                train_ids::Vector{String};
                                polarizations::Vector{String} = ["g1-g1", "g2-g2", "g3-g3"])

    isempty(train_ids)    && error("train_ids is empty")
    isempty(polarizations) && error("polarizations must be non-empty")

    npls = h5open(gauge_h5, "r") do fid
        size(read(fid["configs"][train_ids[1]]["plaq_scalar"]), 5)  # last dim of (L1,L2,L3,L4,npls)
    end

    T = h5open(corr_h5, "r") do fid
        size(read(fid["configs"][train_ids[1]][polarizations[1]]["correlator"]), 1)
    end

    feat_sum  = zeros(Float64, npls)
    feat_sum2 = zeros(Float64, npls)
    feat_n    = zeros(Int64,   npls)

    corr_sum  = zeros(Float64, T)
    corr_sum2 = zeros(Float64, T)
    corr_n    = 0

    h5open(gauge_h5, "r") do gfid
        for cid in train_ids
            ps = read(gfid["configs"][cid]["plaq_scalar"])
            for ipl in 1:npls
                vals = @view ps[:, :, :, :, ipl]
                feat_sum[ipl]  += sum(vals)
                feat_sum2[ipl] += sum(vals .^ 2)
                feat_n[ipl]    += length(vals)
            end
        end
    end

    h5open(corr_h5, "r") do cfid
        for cid in train_ids
            for pol in polarizations
                co   = read(cfid["configs"][cid][pol]["correlator"])
                cbar = vec(mean(co, dims=2))   # source-averaged: Float64[T]
                for t in 1:T
                    corr_sum[t]  += cbar[t]
                    corr_sum2[t] += cbar[t]^2
                end
                corr_n += 1   # one source-averaged sample per (config, polarization)
            end
        end
    end

    feat_mean = feat_sum  ./ feat_n
    feat_var  = feat_sum2 ./ feat_n .- feat_mean .^ 2
    feat_std  = sqrt.(max.(feat_var, 0.0))

    corr_mean = corr_sum  ./ corr_n
    corr_var  = corr_sum2 ./ corr_n .- corr_mean .^ 2
    corr_std  = sqrt.(max.(corr_var, 0.0))

    feat_std[feat_std .< 1e-12] .= 1.0
    corr_std[corr_std .< 1e-12] .= 1.0

    return NormStats(feat_mean, feat_std, corr_mean, corr_std)
end

"""
    save_normalization(h5path, stats::NormStats)

Write normalization statistics into the `normalization/` group of an existing
HDF5 file. Overwrites if the group already exists.
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
    load_gauge(gauge_h5, cid; stats=nothing, field=:scalar) -> features

Load plaquette features for one configuration from a gauge database.

- `field = :scalar` → `Float64[iL[1], iL[2], iL[3], iL[4], npls]`
  Pass the path to the scalar gauge database (`gauge_scalar_db.h5`).
- `field = :matrix` → `ComplexF64[6, iL[1], iL[2], iL[3], iL[4], npls]`
  Pass the path to the matrix gauge database (`gauge_matrix_db.h5`).

If `stats::NormStats` is provided, `plaq_scalar` is z-score normalized per
plane. `plaq_matrix` is never normalized (complex SU3 matrices).
"""
function load_gauge(gauge_h5::String, cid::String;
                    stats::Union{NormStats, Nothing} = nothing,
                    field::Symbol = :scalar)

    field in (:scalar, :matrix) ||
        error("field must be :scalar or :matrix")

    features = h5open(gauge_h5, "r") do fid
        haskey(fid["configs"], cid) ||
            error("Config \"$(cid)\" not found in $(gauge_h5)")
        grp = fid["configs"][cid]
        key = field == :scalar ? "plaq_scalar" : "plaq_matrix"
        haskey(grp, key) ||
            error("\"$(key)\" not found for config \"$(cid)\" in $(gauge_h5). " *
                  "Make sure you are passing the correct database file.")
        read(grp[key])
    end

    stats === nothing && return features
    return _normalize_features(features, stats, field)
end

"""
    load_corr(corr_h5, cid; stats=nothing, polarization="g1-g1") -> correlator

Load the correlator `Float64[T, nsrcs]` for one configuration and one
polarization from the correlator database. Applies z-score normalization
per time slice if `stats::NormStats` is provided.
"""
function load_corr(corr_h5::String, cid::String;
                   stats::Union{NormStats, Nothing} = nothing,
                   polarization::String = "g1-g1")

    correlator = h5open(corr_h5, "r") do fid
        haskey(fid["configs"], cid) ||
            error("Config \"$(cid)\" not found in $(corr_h5)")
        grp = fid["configs"][cid]
        haskey(grp, polarization) ||
            error("Polarization \"$(polarization)\" not found for config \"$(cid)\". " *
                  "Rebuild with the desired polarizations.")
        read(grp[polarization]["correlator"])
    end

    stats === nothing && return correlator
    return _normalize_correlator(correlator, stats)
end

"""
    load_config(gauge_h5, corr_h5, cid;
                stats=nothing, field=:scalar, polarization="g1-g1")
        -> (features, correlator)

Load features and correlator for one configuration from the two-database
design. Combines `load_gauge` and `load_corr`. Pass the appropriate gauge
database path for the chosen `field` (`:scalar` or `:matrix`).
"""
function load_config(gauge_h5::String, corr_h5::String, cid::String;
                     stats::Union{NormStats, Nothing} = nothing,
                     field::Symbol = :scalar,
                     polarization::String = "g1-g1")

    features   = load_gauge(gauge_h5, cid; stats=stats, field=field)
    correlator = load_corr(corr_h5, cid; stats=stats, polarization=polarization)
    return features, correlator
end

"""
    load_split(gauge_h5, corr_h5, ids;
               stats=nothing, field=:scalar, polarization="g1-g1")
        -> (features_list, correlator_list)

Load all configurations in `ids` from the two-database design, in MC order.
Returns two `Vector`s. If `stats` is provided, data is normalized.
"""
function load_split(gauge_h5::String, corr_h5::String, ids::Vector{String};
                    stats::Union{NormStats, Nothing} = nothing,
                    field::Symbol = :scalar,
                    polarization::String = "g1-g1")

    features_list   = Vector{Any}(undef, length(ids))
    correlator_list = Vector{Matrix{Float64}}(undef, length(ids))
    for (i, cid) in enumerate(ids)
        features_list[i], correlator_list[i] =
            load_config(gauge_h5, corr_h5, cid;
                        stats=stats, field=field, polarization=polarization)
    end
    return features_list, correlator_list
end

# ---------------------------------------------------------------------------
# In-memory dataset preloading
# ---------------------------------------------------------------------------

"""
    PreloadedDataset

In-memory cache mapping config id → `(feat, corr2d)` where:
- `feat`   : `Float32[iL1, iL2, iL3, iL4, npls]` — normalised `plaq_scalar`
- `corr2d` : `Float32[Lt, npol]` — source-averaged, normalised correlator
              for all requested polarizations (column order = `polarizations` arg)

Build with `preload_dataset`. Pass to the `load_batch(ids, cache)` overload
in training scripts to avoid per-batch HDF5 reads.
"""
struct PreloadedDataset
    data :: Dict{String, Tuple{Array{Float32, 5}, Matrix{Float32}}}
end

Base.length(d::PreloadedDataset)                = length(d.data)
Base.getindex(d::PreloadedDataset, cid::String) = d.data[cid]
Base.keys(d::PreloadedDataset)                  = keys(d.data)

"""
    preload_dataset(gauge_h5, corr_h5, ids, stats;
                    polarizations = ["g1-g1","g2-g2","g3-g3"]) -> PreloadedDataset

Load and normalise all configs in `ids` into RAM as `Float32` arrays, opening
each HDF5 file only once. Intended for the gauge scalar database; matrix
features are not supported here (too large to preload routinely).

Memory estimate per config on A654 (48×24³×6 Float32): ≈ 24 MB.
For 680 configs: ≈ 16 GB. Use Float32 storage in `build_gauge_dataset` to
keep this within a typical 32–64 GB RAM budget.
"""
function preload_dataset(gauge_h5::String, corr_h5::String, ids::Vector{String},
                         stats::NormStats;
                         polarizations::Vector{String} = ["g1-g1", "g2-g2", "g3-g3"])
    isempty(ids) && return PreloadedDataset(
        Dict{String, Tuple{Array{Float32,5}, Matrix{Float32}}}())

    npol = length(polarizations)
    T    = h5open(corr_h5, "r") do fid
        size(read(fid["configs"][ids[1]][polarizations[1]]["correlator"]), 1)
    end

    data = Dict{String, Tuple{Array{Float32, 5}, Matrix{Float32}}}()
    sizehint!(data, length(ids))

    h5open(gauge_h5, "r") do gfid
        h5open(corr_h5, "r") do cfid
            for cid in ids
                # normalised features
                raw  = read(gfid["configs"][cid]["plaq_scalar"])
                feat = Float32.(_normalize_features(raw, stats, :scalar))

                # source-averaged, normalised correlator
                corr2d = Matrix{Float32}(undef, T, npol)
                for (ipol, pol) in enumerate(polarizations)
                    co   = read(cfid["configs"][cid][pol]["correlator"])
                    cbar = vec(mean(co, dims=2))
                    for t in 1:T
                        corr2d[t, ipol] = Float32(
                            (cbar[t] - stats.corr_mean[t]) / stats.corr_std[t])
                    end
                end

                data[cid] = (feat, corr2d)
            end
        end
    end

    return PreloadedDataset(data)
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
    acc     = zeros(Float64, nsplits)

    for i in 1:N
        acc .+= fracs
        j = argmax(acc)
        labels[i] = j
        acc[j] -= 1.0
    end

    return labels
end

function _normalize_features(features, stats::NormStats, field::Symbol)
    if field == :scalar
        # features shape: (iL[1], iL[2], iL[3], iL[4], npls)
        out = similar(features)
        for ipl in axes(features, 5)
            out[:, :, :, :, ipl] = (features[:, :, :, :, ipl] .- stats.feat_mean[ipl]) ./
                                    stats.feat_std[ipl]
        end
        return out
    else  # :matrix — complex SU3 entries, no normalization
        return features
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
