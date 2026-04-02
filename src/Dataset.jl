module Dataset

using HDF5
using LatticeGPU

import ..IO: import_cern64
import ..Plaquette: plaquette_scalar_field, plaquette_field

export build_gauge_dataset, build_gauge_matrix_dataset, build_corr_dataset, merge_dataset

import ..Correlator: get_LMAConfig_all_sources

# ---------------------------------------------------------------------------
# HDF5 layouts
# ---------------------------------------------------------------------------
#
#   gauge_scalar_db.h5
#   ├── metadata/
#   │   ├── vol          Int64[4]   — full lattice extent (iL)
#   │   ├── svol         Int64[4]   — block size (blk)
#   │   ├── ensemble     String     — path to gauge config directory
#   │   └── config_fmt   String     — gauge reader format
#   └── configs/
#       └── <id>/
#           └── plaq_scalar   Float64[iL[1], iL[2], iL[3], iL[4], npls]
#                               — Re(Tr P_μν(x)) in full spatial layout
#
#   gauge_matrix_db.h5
#   ├── metadata/
#   │   ├── vol, svol, ensemble, config_fmt   (same as above)
#   └── configs/
#       └── <id>/
#           └── plaq_matrix   ComplexF64[6, iL[1], iL[2], iL[3], iL[4], npls]
#                               — first 2 rows of untraced P_μν(x), spatial layout
#                               — dim 1: [u11, u12, u13, u21, u22, u23]
#
#   corr_db.h5
#   ├── metadata/
#   │   ├── lma_path      String
#   │   ├── em            String
#   │   └── polarizations String[npols]
#   └── configs/
#       └── <id>/
#           └── <polarization>/
#               ├── correlator   Float64[T, nsrcs]
#               └── sources      String[nsrcs]

# ---------------------------------------------------------------------------
# build_gauge_dataset  (scalar plaquette only)
# ---------------------------------------------------------------------------

"""
    build_gauge_dataset(ensemble_path, lp, output_path;
                        config_fmt   = "cern",
                        config_range = nothing,
                        verbose      = true)

Build the **scalar** gauge-features HDF5 database from gauge configurations in
`ensemble_path`. Stores only `plaq_scalar = Re(Tr P_μν(x))` in full spatial
layout `Float64[iL[1], iL[2], iL[3], iL[4], npls]`, ready for direct use in
the CNN without any further reconstruction.

`config_range` selects a sub-range of configs by position in the sorted list
(e.g. `config_range = 1:100` → first 100, `101:200` → next 100). Pass
`nothing` (default) to process all available configs.

Use alongside `build_gauge_matrix_dataset` if you also need the SU(3) matrix
features. The two databases share the same config id keys and can be processed
independently or in separate jobs.

# HDF5 schema
```
gauge_scalar_db.h5
├── metadata/  vol, svol, ensemble, config_fmt
└── configs/
    └── <id>/
        └── plaq_scalar  Float64[iL[1], iL[2], iL[3], iL[4], npls]
```

Config ids are the integers after `n` in gauge filenames (e.g. `A654r000n1` → `"1"`).
"""
function build_gauge_dataset(ensemble_path::String,
                              lp,
                              output_path::String;
                              config_fmt   :: String                          = "cern",
                              config_range :: Union{UnitRange{Int}, Nothing}  = nothing,
                              verbose      :: Bool                             = true)

    cfg_ids = _gauge_config_ids(ensemble_path, config_range, verbose)
    gauge_map = _gauge_map(ensemble_path)
    reader = _make_reader(config_fmt, lp)

    verbose && @info "Building gauge scalar dataset: $(length(cfg_ids)) configs → $(output_path)"

    h5open(output_path, "w") do fid
        _write_gauge_metadata(fid, lp, ensemble_path, config_fmt)
        cfgs_grp = create_group(fid, "configs")

        for (i, cid) in enumerate(cfg_ids)
            verbose && @info "  [$i/$(length(cfg_ids))] config $(cid)"
            U  = reader(joinpath(ensemble_path, gauge_map[cid]))
            ps = plaquette_scalar_field(U, lp)
            grp = create_group(cfgs_grp, cid)
            write(grp, "plaq_scalar", _to_spatial_scalar(ps, lp))
        end
    end

    verbose && @info "Gauge scalar dataset written to $(output_path)"
    return output_path
end

# ---------------------------------------------------------------------------
# build_gauge_matrix_dataset  (SU(3) matrix plaquette only)
# ---------------------------------------------------------------------------

"""
    build_gauge_matrix_dataset(ensemble_path, lp, output_path;
                               config_fmt   = "cern",
                               config_range = nothing,
                               verbose      = true)

Build the **matrix** gauge-features HDF5 database from gauge configurations in
`ensemble_path`. Stores only `plaq_matrix` (first 2 rows of the untraced SU(3)
plaquette) in full spatial layout `ComplexF64[6, iL[1], iL[2], iL[3], iL[4], npls]`.
The third row is recoverable from SU(3) unitarity.

`config_range` selects a sub-range of configs by position in the sorted list.

# HDF5 schema
```
gauge_matrix_db.h5
├── metadata/  vol, svol, ensemble, config_fmt
└── configs/
    └── <id>/
        └── plaq_matrix  ComplexF64[6, iL[1], iL[2], iL[3], iL[4], npls]
```
"""
function build_gauge_matrix_dataset(ensemble_path::String,
                                     lp,
                                     output_path::String;
                                     config_fmt   :: String                          = "cern",
                                     config_range :: Union{UnitRange{Int}, Nothing}  = nothing,
                                     verbose      :: Bool                             = true)

    cfg_ids = _gauge_config_ids(ensemble_path, config_range, verbose)
    gauge_map = _gauge_map(ensemble_path)
    reader = _make_reader(config_fmt, lp)

    verbose && @info "Building gauge matrix dataset: $(length(cfg_ids)) configs → $(output_path)"

    h5open(output_path, "w") do fid
        _write_gauge_metadata(fid, lp, ensemble_path, config_fmt)
        cfgs_grp = create_group(fid, "configs")

        for (i, cid) in enumerate(cfg_ids)
            verbose && @info "  [$i/$(length(cfg_ids))] config $(cid)"
            U   = reader(joinpath(ensemble_path, gauge_map[cid]))
            pf  = plaquette_field(U, lp)
            grp = create_group(cfgs_grp, cid)
            write(grp, "plaq_matrix", _to_spatial_matrix(pf, lp))
        end
    end

    verbose && @info "Gauge matrix dataset written to $(output_path)"
    return output_path
end

# ---------------------------------------------------------------------------
# build_corr_dataset
# ---------------------------------------------------------------------------

"""
    build_corr_dataset(lma_path, output_path;
                       em            = "VV",
                       polarizations = ["g1-g1", "g2-g2", "g3-g3"],
                       config_range  = nothing,
                       verbose       = true)

Build a correlator HDF5 database from the LMA rest-eigen correlators in
`lma_path`, storing all requested polarizations for each configuration.

`config_range` selects a sub-range of configs by position in the sorted list.

# HDF5 schema
```
corr_db.h5
├── metadata/  lma_path, em, polarizations
└── configs/
    └── <id>/
        └── <polarization>/
            ├── correlator  Float64[T, nsrcs]
            └── sources     String[nsrcs]
```

Config ids are plain integer names of subdirectories under `lma_path`.
"""
function build_corr_dataset(lma_path::String,
                             output_path::String;
                             em            :: String                          = "VV",
                             polarizations :: Vector{String}                  = ["g1-g1", "g2-g2", "g3-g3"],
                             config_range  :: Union{UnitRange{Int}, Nothing}  = nothing,
                             verbose       :: Bool                             = true)

    isdir(lma_path) || error("LMA directory not found: $(lma_path)")
    isempty(polarizations) && error("polarizations must be non-empty")

    lma_ids = sort(filter(f -> isdir(joinpath(lma_path, f)) && _is_integer(f),
                          readdir(lma_path)),
                   by = x -> parse(Int, x))
    isempty(lma_ids) && error("No LMA config directories found in $(lma_path).")

    if config_range !== nothing
        _check_range(config_range, length(lma_ids))
        lma_ids = lma_ids[config_range]
        verbose && @info "config_range=$(config_range): processing $(length(lma_ids)) configs"
    end

    verbose && @info "Building correlator dataset: $(length(lma_ids)) configs → $(output_path)"

    h5open(output_path, "w") do fid
        meta = create_group(fid, "metadata")
        write(meta, "lma_path",      lma_path)
        write(meta, "em",            em)
        write(meta, "polarizations", polarizations)

        cfgs_grp = create_group(fid, "configs")

        for (i, cid) in enumerate(lma_ids)
            verbose && @info "  [$i/$(length(lma_ids))] config $(cid)"

            lma_dir = joinpath(lma_path, cid)
            grp     = create_group(cfgs_grp, cid)

            for pol in polarizations
                lcfg    = get_LMAConfig_all_sources(lma_dir, pol; em=em, re_only=true)
                re_dict = lcfg.data["re"]

                sources    = collect(keys(re_dict))
                T          = length(re_dict[sources[1]])
                nsrcs      = length(sources)
                correlator = Matrix{Float64}(undef, T, nsrcs)
                for (j, src) in enumerate(sources)
                    correlator[:, j] = re_dict[src]
                end

                pol_grp = create_group(grp, pol)
                write(pol_grp, "correlator", correlator)
                write(pol_grp, "sources",    sources)
            end
        end
    end

    verbose && @info "Correlator dataset written to $(output_path)"
    return output_path
end

# ---------------------------------------------------------------------------
# merge_dataset
# ---------------------------------------------------------------------------

"""
    merge_dataset(input_paths, output_path; verbose = true)

Merge multiple HDF5 dataset shards (produced by `build_gauge_dataset`,
`build_gauge_matrix_dataset`, or `build_corr_dataset` with `config_range`)
into a single file.

Works for **all three database types** — the function copies every
`configs/<id>` group verbatim, so the schema inside each config group is
preserved regardless of whether it contains `plaq_scalar`, `plaq_matrix`,
or correlator sub-groups.

Metadata is taken from the first input file.  For gauge databases the
`vol` and `svol` arrays are checked for consistency across all shards;
a mismatch raises an error.  Duplicate config ids across shards also raise
an error.

# Example
```julia
merge_dataset(["part1.h5", "part2.h5", "part3.h5"], "merged.h5")
```
"""
function merge_dataset(input_paths::Vector{String},
                       output_path::String;
                       verbose::Bool = true)

    isempty(input_paths) && error("input_paths must be non-empty")
    for p in input_paths
        isfile(p) || error("Input file not found: $(p)")
    end

    verbose && @info "Merging $(length(input_paths)) shards → $(output_path)"

    # ------------------------------------------------------------------
    # Read reference metadata from the first shard
    # ------------------------------------------------------------------
    ref_meta = h5open(input_paths[1], "r") do fid
        _read_all_metadata(fid)
    end

    # ------------------------------------------------------------------
    # Collect all config ids across shards; check for duplicates
    # ------------------------------------------------------------------
    shard_ids = Vector{Vector{String}}(undef, length(input_paths))
    for (k, p) in enumerate(input_paths)
        h5open(p, "r") do fid
            shard_ids[k] = haskey(fid, "configs") ? keys(fid["configs"]) : String[]
        end
    end

    seen    = Dict{String, Int}()   # id → shard index (1-based) of first occurrence
    for (k, ids) in enumerate(shard_ids)
        for id in ids
            if haskey(seen, id)
                error("Duplicate config id \"$(id)\" found in shard $(k) " *
                      "and shard $(seen[id]).")
            end
            seen[id] = k
        end
    end
    total = sum(length, shard_ids)
    verbose && @info "  Found $(total) configs across shards (no duplicates)"

    # ------------------------------------------------------------------
    # Write merged file
    # ------------------------------------------------------------------
    h5open(output_path, "w") do out_fid

        # Copy metadata from first shard
        h5open(input_paths[1], "r") do src_fid
            if haskey(src_fid, "metadata")
                HDF5.copy_object(src_fid["metadata"], out_fid, "metadata")
            end
        end

        cfgs_out = create_group(out_fid, "configs")

        for (k, p) in enumerate(input_paths)
            verbose && @info "  Shard $k/$(length(input_paths)): $(p)"
            h5open(p, "r") do src_fid

                # Validate metadata consistency for gauge databases (vol/svol)
                if k > 1
                    _check_metadata_consistency(ref_meta,
                                                _read_all_metadata(src_fid),
                                                p)
                end

                !haskey(src_fid, "configs") && return

                for cid in shard_ids[k]
                    verbose && @info "    config $(cid)"
                    HDF5.copy_object(src_fid["configs"][cid], cfgs_out, cid)
                end
            end
        end
    end

    verbose && @info "Merged dataset written to $(output_path)"
    return output_path
end

# Read all scalar/vector metadata entries into a flat Dict for comparison.
function _read_all_metadata(fid)
    d = Dict{String, Any}()
    haskey(fid, "metadata") || return d
    for k in keys(fid["metadata"])
        d[k] = read(fid["metadata"][k])
    end
    return d
end

# Only check array-valued keys that must agree across shards (vol, svol).
const _ARRAY_META_KEYS = ("vol", "svol")

function _check_metadata_consistency(ref::Dict, other::Dict, path::String)
    for k in _ARRAY_META_KEYS
        haskey(ref, k) || continue
        haskey(other, k) || error("Metadata key \"$(k)\" missing in $(path).")
        ref[k] == other[k] ||
            error("Metadata mismatch for key \"$(k)\" in $(path): " *
                  "expected $(ref[k]), got $(other[k]).")
    end
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_is_integer(s::String) = !isnothing(tryparse(Int, s))

function _config_id(filename::String)
    m = match(r"n(\d+)$", filename)
    m === nothing ? nothing : m.captures[1]
end

function _gauge_map(ensemble_path::String)
    isdir(ensemble_path) || error("Gauge config directory not found: $(ensemble_path)")
    m = Dict{String,String}()
    for f in readdir(ensemble_path)
        isdir(joinpath(ensemble_path, f)) && continue
        id = _config_id(f)
        id === nothing || (m[id] = f)
    end
    isempty(m) && error("No gauge configs found in $(ensemble_path). " *
                         "Filenames must contain a trailing n<integer> (e.g. A654r000n1).")
    return m
end

function _gauge_config_ids(ensemble_path::String,
                            config_range::Union{UnitRange{Int}, Nothing},
                            verbose::Bool)
    ids = sort(collect(keys(_gauge_map(ensemble_path))), by = x -> parse(Int, x))
    if config_range !== nothing
        _check_range(config_range, length(ids))
        ids = ids[config_range]
        verbose && @info "config_range=$(config_range): processing $(length(ids)) configs"
    end
    return ids
end

function _check_range(r::UnitRange{Int}, n::Int)
    (first(r) >= 1 && last(r) <= n) ||
        error("config_range=$(r) is out of bounds for $(n) available configs.")
end

function _write_gauge_metadata(fid, lp, ensemble_path::String, config_fmt::String)
    meta = create_group(fid, "metadata")
    write(meta, "vol",        collect(Int64, lp.iL))
    write(meta, "svol",       collect(Int64, lp.blk))
    write(meta, "ensemble",   ensemble_path)
    write(meta, "config_fmt", config_fmt)
end

function _make_reader(fmt::String, lp)
    fmt == "cern" && return path -> import_cern64(path, 0, lp; log=false)
    error("Unsupported config format \"$(fmt)\". Only \"cern\" is currently supported.")
end

"""
    _to_spatial_scalar(ps, lp) -> Array{Float64, 5}

Convert block-decomposed `(bsz, npls, rsz)` scalar plaquette array to full
spatial layout `(iL[1], iL[2], iL[3], iL[4], npls)` using `point_coord`.
"""
function _to_spatial_scalar(ps::Array{<:Real, 3}, lp)
    npls = size(ps, 2)
    out  = Array{Float64, 5}(undef, lp.iL[1], lp.iL[2], lp.iL[3], lp.iL[4], npls)
    for r in 1:lp.rsz, b in 1:lp.bsz
        coord = point_coord((b, r), lp)
        i1, i2, i3, i4 = coord[1], coord[2], coord[3], coord[4]
        for ipl in 1:npls
            out[i1, i2, i3, i4, ipl] = ps[b, ipl, r]
        end
    end
    return out
end

"""
    _to_spatial_matrix(pf, lp) -> Array{ComplexF64, 6}

Convert block-decomposed `(bsz, npls, rsz)` SU3 plaquette array to full
spatial layout `(6, iL[1], iL[2], iL[3], iL[4], npls)` using `point_coord`.
Stores the 6 independent complex entries of the first two rows.
"""
function _to_spatial_matrix(pf::Array{SU3{T}, 3}, lp) where {T}
    npls = size(pf, 2)
    out  = Array{Complex{T}, 6}(undef, 6, lp.iL[1], lp.iL[2], lp.iL[3], lp.iL[4], npls)
    for r in 1:lp.rsz, b in 1:lp.bsz
        coord = point_coord((b, r), lp)
        i1, i2, i3, i4 = coord[1], coord[2], coord[3], coord[4]
        for ipl in 1:npls
            p = pf[b, ipl, r]
            out[1, i1, i2, i3, i4, ipl] = p.u11
            out[2, i1, i2, i3, i4, ipl] = p.u12
            out[3, i1, i2, i3, i4, ipl] = p.u13
            out[4, i1, i2, i3, i4, ipl] = p.u21
            out[5, i1, i2, i3, i4, ipl] = p.u22
            out[6, i1, i2, i3, i4, ipl] = p.u23
        end
    end
    return out
end

end # module Dataset
