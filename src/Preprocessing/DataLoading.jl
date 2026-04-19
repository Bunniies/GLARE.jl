# ---------------------------------------------------------------------------
# Per-config loaders
# ---------------------------------------------------------------------------

"""
    load_links(links_h5, cid) -> ComplexF32[6, Lt, Ls, Ls, Ls, ndim]

Load raw gauge link storage for one configuration from a gauge link database.
Returns the first-two-row storage format as-is (no SU(3) reconstruction).
Layout: `(6, iL[4], iL[1], iL[2], iL[3], ndim)` — temporal coordinate first
(dim2=Lt) so that after `su3_reconstruct` the array is `(3,3,Lt,Ls,Ls,Ls,ndim)`.
Call `su3_reconstruct` on the result to obtain full 3×3 matrices.
"""
function load_links(links_h5::String, cid::String)
    @timeit GLARE_TIMER "load_links" begin
    h5open(links_h5, "r") do fid
        haskey(fid["configs"], cid) ||
            error("Config \"$(cid)\" not found in $(links_h5)")
        grp = fid["configs"][cid]
        haskey(grp, "gauge_links") ||
            error("\"gauge_links\" not found for config \"$(cid)\" in $(links_h5). " *
                  "Make sure you are passing the gauge link database.")
        read(grp["gauge_links"])   # ComplexF32[6, iL4, iL1, iL2, iL3, ndim] = [6, Lt, Ls, Ls, Ls, ndim]
    end
    end # timeit
end

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
    @timeit GLARE_TIMER "load_gauge" begin
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
    end # timeit
end

"""
    load_corr(corr_h5, cid; stats=nothing, polarization="g1-g1") -> correlator

Load the correlator `Float64[T, nsrcs]` for one configuration and one
polarization. Applies z-score normalization per time slice if `stats` is provided.
"""
function load_corr(corr_h5::String, cid::String;
                   stats::Union{NormStats, Nothing} = nothing,
                   polarization::String = "g1-g1")
    @timeit GLARE_TIMER "load_corr" begin
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
    end # timeit
end

"""
    load_config(gauge_h5, corr_h5, cid;
                stats=nothing, field=:scalar, polarization="g1-g1")
        -> (features, correlator)

Load features and correlator for one configuration. Combines `load_gauge` and
`load_corr`. Pass the appropriate gauge database path for the chosen `field`.
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

Load all configurations in `ids` in MC order. Returns two `Vector`s.
If `stats` is provided, data is normalized.
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
    @timeit GLARE_TIMER "preload_dataset" begin
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
                raw  = read(gfid["configs"][cid]["plaq_scalar"])
                feat = Float32.(_normalize_features(raw, stats, :scalar))

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
    end # timeit
end
