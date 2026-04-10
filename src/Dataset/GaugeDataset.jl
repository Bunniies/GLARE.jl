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
#
#   gauge_matrix_db.h5
#   ├── metadata/  (same as above)
#   └── configs/
#       └── <id>/
#           └── plaq_matrix   ComplexF64[6, iL[1], iL[2], iL[3], iL[4], npls]
#                               — first 2 rows of untraced P_μν(x)
#
#   gauge_links_db.h5
#   ├── metadata/  (same as above)
#   └── configs/
#       └── <id>/
#           └── gauge_links   ComplexF32[6, iL[1], iL[2], iL[3], iL[4], ndim]
#                               — first 2 rows of each link U_μ(x), dim 6 = ndim=4

# ---------------------------------------------------------------------------
# build_gauge_dataset  (scalar plaquette)
# ---------------------------------------------------------------------------

"""
    build_gauge_dataset(ensemble_path, lp, output_path;
                        config_fmt   = "cern",
                        config_range = nothing,
                        verbose      = true)

Build the **scalar** gauge-features HDF5 database from gauge configurations in
`ensemble_path`. Stores only `plaq_scalar = Re(Tr P_μν(x))` in full spatial
layout `Float64[iL[1], iL[2], iL[3], iL[4], npls]`.

`config_range` selects a sub-range of configs by position in the sorted list
(e.g. `config_range = 1:100`). Pass `nothing` to process all configs.

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

    cfg_ids   = _gauge_config_ids(ensemble_path, config_range, verbose)
    gauge_map = _gauge_map(ensemble_path)
    reader    = _make_reader(config_fmt, lp)

    verbose && @info "Building gauge scalar dataset: $(length(cfg_ids)) configs → $(output_path)"

    h5open(output_path, "w") do fid
        _write_gauge_metadata(fid, lp, ensemble_path, config_fmt)
        cfgs_grp = create_group(fid, "configs")

        for (i, cid) in enumerate(cfg_ids)
            verbose && @info "  [$i/$(length(cfg_ids))] config $(cid)"
            U   = reader(joinpath(ensemble_path, gauge_map[cid]))
            ps  = plaquette_scalar_field(U, lp)
            grp = create_group(cfgs_grp, cid)
            write(grp, "plaq_scalar", _to_spatial_scalar(ps, lp))
        end
    end

    verbose && @info "Gauge scalar dataset written to $(output_path)"
    return output_path
end

# ---------------------------------------------------------------------------
# build_gauge_matrix_dataset  (SU(3) matrix plaquette)
# ---------------------------------------------------------------------------

"""
    build_gauge_matrix_dataset(ensemble_path, lp, output_path;
                               config_fmt   = "cern",
                               config_range = nothing,
                               verbose      = true)

Build the **matrix** gauge-features HDF5 database. Stores `plaq_matrix`
(first 2 rows of the untraced SU(3) plaquette) in full spatial layout
`ComplexF64[6, iL[1], iL[2], iL[3], iL[4], npls]`.

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

    cfg_ids   = _gauge_config_ids(ensemble_path, config_range, verbose)
    gauge_map = _gauge_map(ensemble_path)
    reader    = _make_reader(config_fmt, lp)

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
# build_gauge_link_dataset  (raw SU(3) link matrices)
# ---------------------------------------------------------------------------

"""
    build_gauge_link_dataset(ensemble_path, lp, output_path;
                             config_fmt   = "cern",
                             config_range = nothing,
                             verbose      = true)

Build the **gauge link** HDF5 database. Stores `gauge_links` (first 2 rows of
each link matrix `U_μ(x)`) as `ComplexF32[6, iL[1], iL[2], iL[3], iL[4], ndim]`.

Links are stored as `ComplexF32` (half the disk/RAM of `ComplexF64`) — the
SU(3) constraint fixes precision to ~1e-7 per element after reconstruction,
sufficient for Float32. Configs are read in Float64 and converted on write.

The third SU(3) row is recoverable via `su3_reconstruct`. Primary input
database for the Phase 2 L-CNN.

`config_range` selects a sub-range of configs by position in the sorted list.

# HDF5 schema
```
gauge_links_db.h5
├── metadata/  vol, svol, ensemble, config_fmt
└── configs/
    └── <id>/
        └── gauge_links  ComplexF32[6, iL[1], iL[2], iL[3], iL[4], ndim]
```

Dim 1 stores `[u11, u12, u13, u21, u22, u23]` (first two rows).
Dim 6 indexes the 4 spacetime directions μ (LatticeGPU order: x, y, z, t).
"""
function build_gauge_link_dataset(ensemble_path::String,
                                   lp,
                                   output_path::String;
                                   config_fmt   :: String                          = "cern",
                                   config_range :: Union{UnitRange{Int}, Nothing}  = nothing,
                                   verbose      :: Bool                             = true)

    cfg_ids   = _gauge_config_ids(ensemble_path, config_range, verbose)
    gauge_map = _gauge_map(ensemble_path)
    reader    = _make_reader(config_fmt, lp)

    verbose && @info "Building gauge link dataset: $(length(cfg_ids)) configs → $(output_path)"

    h5open(output_path, "w") do fid
        _write_gauge_metadata(fid, lp, ensemble_path, config_fmt)
        cfgs_grp = create_group(fid, "configs")

        for (i, cid) in enumerate(cfg_ids)
            verbose && @info "  [$i/$(length(cfg_ids))] config $(cid)"
            U   = reader(joinpath(ensemble_path, gauge_map[cid]))
            grp = create_group(cfgs_grp, cid)
            write(grp, "gauge_links", ComplexF32.(_to_spatial_links(U, lp)))
        end
    end

    verbose && @info "Gauge link dataset written to $(output_path)"
    return output_path
end

# ---------------------------------------------------------------------------
# Internal layout converters
# ---------------------------------------------------------------------------

"""
    _to_spatial_scalar(ps, lp) -> Array{Float64, 5}

Convert block-decomposed `(bsz, npls, rsz)` scalar plaquette array to full
spatial layout `(Lt, Ls, Ls, Ls, npls)` = `(iL[4], iL[1], iL[2], iL[3], npls)`
using `point_coord`. The temporal coordinate (coord4, period `iL[4]`) is placed
first so that dim1=Lt matches the `(Lt, Ls, Ls, Ls, ...)` GLARE array convention.
"""
function _to_spatial_scalar(ps::Array{<:Real, 3}, lp)
    npls = size(ps, 2)
    out  = Array{Float64, 5}(undef, lp.iL[4], lp.iL[1], lp.iL[2], lp.iL[3], npls)
    for r in 1:lp.rsz, b in 1:lp.bsz
        coord = point_coord((b, r), lp)
        i1, i2, i3, i4 = coord[1], coord[2], coord[3], coord[4]
        for ipl in 1:npls
            out[i4, i1, i2, i3, ipl] = ps[b, ipl, r]
        end
    end
    return out
end

"""
    _to_spatial_matrix(pf, lp) -> Array{ComplexF64, 6}

Convert block-decomposed `(bsz, npls, rsz)` SU3 plaquette array to full
spatial layout `(6, Lt, Ls, Ls, Ls, npls)` = `(6, iL[4], iL[1], iL[2], iL[3], npls)`
using `point_coord`. Temporal coordinate placed first (dim2=Lt).
Stores the 6 independent complex entries of the first two rows.
"""
function _to_spatial_matrix(pf::Array{SU3{T}, 3}, lp) where {T}
    npls = size(pf, 2)
    out  = Array{Complex{T}, 6}(undef, 6, lp.iL[4], lp.iL[1], lp.iL[2], lp.iL[3], npls)
    for r in 1:lp.rsz, b in 1:lp.bsz
        coord = point_coord((b, r), lp)
        i1, i2, i3, i4 = coord[1], coord[2], coord[3], coord[4]
        for ipl in 1:npls
            p = pf[b, ipl, r]
            out[1, i4, i1, i2, i3, ipl] = p.u11
            out[2, i4, i1, i2, i3, ipl] = p.u12
            out[3, i4, i1, i2, i3, ipl] = p.u13
            out[4, i4, i1, i2, i3, ipl] = p.u21
            out[5, i4, i1, i2, i3, ipl] = p.u22
            out[6, i4, i1, i2, i3, ipl] = p.u23
        end
    end
    return out
end

"""
    _to_spatial_links(U, lp) -> Array{ComplexF64, 6}

Convert block-decomposed `(bsz, ndim, rsz)` link array to full spatial layout
`(6, Lt, Ls, Ls, Ls, ndim)` = `(6, iL[4], iL[1], iL[2], iL[3], ndim)` using
`point_coord`. The temporal coordinate (coord4, period `iL[4]=Lt`) is placed at
dim2 so that dim2=Lt, matching the `(3,3,Lt,Ls,Ls,Ls,...)` GLARE array convention
after `su3_reconstruct`. This makes `_ddir(4)=3` correct for `plaquette_matrices`
and `GaugeEquivConv`.
Stores the 6 independent complex entries of the first two rows of each U_μ(x).
"""
function _to_spatial_links(U::Array{SU3{T}, 3}, lp) where {T}
    ndim = size(U, 2)
    out  = Array{Complex{T}, 6}(undef, 6, lp.iL[4], lp.iL[1], lp.iL[2], lp.iL[3], ndim)
    for r in 1:lp.rsz, b in 1:lp.bsz
        coord = point_coord((b, r), lp)
        i1, i2, i3, i4 = coord[1], coord[2], coord[3], coord[4]
        for iμ in 1:ndim
            u = U[b, iμ, r]
            out[1, i4, i1, i2, i3, iμ] = u.u11
            out[2, i4, i1, i2, i3, iμ] = u.u12
            out[3, i4, i1, i2, i3, iμ] = u.u13
            out[4, i4, i1, i2, i3, iμ] = u.u21
            out[5, i4, i1, i2, i3, iμ] = u.u22
            out[6, i4, i1, i2, i3, iμ] = u.u23
        end
    end
    return out
end
