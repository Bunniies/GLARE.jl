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
