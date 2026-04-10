module Dataset

using HDF5
using LatticeGPU

import ..IO: import_cern64
import ..Plaquette: plaquette_scalar_field, plaquette_field
import ..Correlator: get_LMAConfig_all_sources

export build_gauge_dataset, build_gauge_matrix_dataset, build_gauge_link_dataset,
       build_corr_dataset, merge_dataset

# Shared private helpers used by GaugeDataset, CorrDataset, and Merge.
# Must be defined before the includes.

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

include("GaugeDataset.jl")
include("CorrDataset.jl")
include("Merge.jl")

end # module Dataset
