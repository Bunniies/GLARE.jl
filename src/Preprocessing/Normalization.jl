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

Correlator stats are computed on the **source-averaged** correlator `C̄(t)` to
match the training target, so that the normalized targets have unit variance.

**Never uses val/test/bc data.**
"""
function compute_normalization(gauge_h5::String, corr_h5::String,
                                train_ids::Vector{String};
                                polarizations::Vector{String} = ["g1-g1", "g2-g2", "g3-g3"])

    isempty(train_ids)     && error("train_ids is empty")
    isempty(polarizations) && error("polarizations must be non-empty")

    npls = h5open(gauge_h5, "r") do fid
        size(read(fid["configs"][train_ids[1]]["plaq_scalar"]), 5)
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
                cbar = vec(mean(co, dims=2))
                for t in 1:T
                    corr_sum[t]  += cbar[t]
                    corr_sum2[t] += cbar[t]^2
                end
                corr_n += 1
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
    compute_corr_normalization(corr_h5, train_ids;
                               polarizations = ["g1-g1", "g2-g2", "g3-g3"]) -> NormStats

Compute per-time-slice correlator statistics from the training split only.
No gauge database is required — use this for Phase 2 (L-CNN) training where
gauge links are not normalized.

`feat_mean` and `feat_std` in the returned `NormStats` are empty vectors —
passing this object to any function expecting gauge feature normalization
will error, which is the intended behaviour.

Correlator stats are computed on the **source-averaged** `C̄(t)` to match
the training target. **Never uses val/test/bc data.**
"""
function compute_corr_normalization(corr_h5::String,
                                     train_ids::Vector{String};
                                     polarizations::Vector{String} = ["g1-g1", "g2-g2", "g3-g3"])

    isempty(train_ids)     && error("train_ids is empty")
    isempty(polarizations) && error("polarizations must be non-empty")

    T = h5open(corr_h5, "r") do fid
        size(read(fid["configs"][train_ids[1]][polarizations[1]]["correlator"]), 1)
    end

    corr_sum  = zeros(Float64, T)
    corr_sum2 = zeros(Float64, T)
    corr_n    = 0

    h5open(corr_h5, "r") do cfid
        for cid in train_ids
            for pol in polarizations
                co   = read(cfid["configs"][cid][pol]["correlator"])
                cbar = vec(mean(co, dims=2))
                for t in 1:T
                    corr_sum[t]  += cbar[t]
                    corr_sum2[t] += cbar[t]^2
                end
                corr_n += 1
            end
        end
    end

    corr_mean = corr_sum  ./ corr_n
    corr_var  = corr_sum2 ./ corr_n .- corr_mean .^ 2
    corr_std  = sqrt.(max.(corr_var, 0.0))
    corr_std[corr_std .< 1e-12] .= 1.0

    return NormStats(Float64[], Float64[], corr_mean, corr_std)
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
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _interleaved_assign(N, fracs) -> Vector{Int}

Assign each of N items to one of `length(fracs)` splits using a generalised
Bresenham / DDA algorithm. Each item is assigned to the split whose accumulated
deficit is largest, spreading each split uniformly without shuffling.

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
