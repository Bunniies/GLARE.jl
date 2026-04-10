# ---------------------------------------------------------------------------
# merge_dataset
# ---------------------------------------------------------------------------

"""
    merge_dataset(input_paths, output_path; verbose = true)

Merge multiple HDF5 dataset shards (produced by `build_gauge_dataset`,
`build_gauge_link_dataset`, or `build_corr_dataset` with `config_range`)
into a single file.

Works for **all database types** — the function copies every `configs/<id>`
group verbatim, preserving the schema regardless of whether it contains
`plaq_scalar`, `gauge_links`, or correlator sub-groups.

Metadata is taken from the first input file. For gauge databases the `vol` and
`svol` arrays are checked for consistency across all shards; a mismatch raises
an error. Duplicate config ids across shards also raise an error.

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

    ref_meta = h5open(input_paths[1], "r") do fid
        _read_all_metadata(fid)
    end

    shard_ids = Vector{Vector{String}}(undef, length(input_paths))
    for (k, p) in enumerate(input_paths)
        h5open(p, "r") do fid
            shard_ids[k] = haskey(fid, "configs") ? keys(fid["configs"]) : String[]
        end
    end

    seen = Dict{String, Int}()
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

    h5open(output_path, "w") do out_fid

        h5open(input_paths[1], "r") do src_fid
            if haskey(src_fid, "metadata")
                HDF5.copy_object(src_fid["metadata"], out_fid, "metadata")
            end
        end

        cfgs_out = create_group(out_fid, "configs")

        for (k, p) in enumerate(input_paths)
            verbose && @info "  Shard $k/$(length(input_paths)): $(p)"
            h5open(p, "r") do src_fid

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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

function _read_all_metadata(fid)
    d = Dict{String, Any}()
    haskey(fid, "metadata") || return d
    for k in keys(fid["metadata"])
        d[k] = read(fid["metadata"][k])
    end
    return d
end

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
