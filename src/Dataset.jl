module Dataset

using HDF5
using LatticeGPU

import ..IO: import_cern64
import ..Plaquette: plaquette_scalar_field, plaquette_field
import ..Correlator: get_LMAConfig_all_sources

export build_dataset

# ---------------------------------------------------------------------------
# HDF5 layout (written by build_dataset)
# ---------------------------------------------------------------------------
#
#   dataset.h5
#   ├── metadata/
#   │   ├── vol          Int64[4]    — full lattice extent (VOL)
#   │   ├── svol         Int64[4]    — sub-volume / block size (SVOL)
#   │   ├── gamma        String      — gamma structure used for LMA
#   │   ├── em           String      — eigenmode set ("PA" or "VV")
#   │   ├── ensemble     String      — path to gauge config directory
#   │   ├── lma_path     String      — path to LMA data directory
#   │   └── save_matrix  Int64       — 1 if plaq_matrix was written, 0 otherwise
#   └── configs/
#       ├── 1/
#       │   ├── plaq_scalar   Float64[bsz, npls, rsz]     — Re(Tr P_μν(x))
#       │   ├── plaq_matrix   ComplexF64[6, bsz, npls, rsz]  [only if save_matrix=true]
#       │   │                   — untraced P_μν(x): 6 complex entries (first 2 rows of SU3)
#       │   │                   — dim 1: [u11, u12, u13, u21, u22, u23]
#       │   │                   — third row implicit: u3i = conj(u1j*u2k - u1k*u2j)
#       │   ├── correlator    Float64[T, nsrcs]  — re contribution per source
#       │   └── sources       String[nsrcs]      — source positions ("0","12",...)
#       ├── 2/
#       │   └── ...
#       └── ...

# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------

"""
    build_dataset(ensemble_path, lma_path, lp, output_path;
                  gamma       = "g5-g5",
                  em          = "VV",
                  config_fmt  = "cern",
                  save_matrix = false,
                  verbose     = true)

Build an HDF5 dataset pairing gauge-field features with rest-eigen correlator
targets for all configurations found in both `ensemble_path` and `lma_path`.

Directory layout expected:

    ensemble_path/
      A654r000n1      ← CERN-format gauge config; config id is the integer after 'n'
      A654r000n2
      ...

    lma_path/
      1/              ← subdirectory named by the plain config integer
        mseig<em>re.dat
      2/
        ...

Config ids are matched by the integer suffix after `n` in gauge filenames
(e.g. `A654r000n1` → id `"1"`) against plain-integer LMA subdirectory names.
The two directories are independent and can live anywhere on disk.
Only configurations present in **both** are processed; a warning is emitted
for any mismatch.

# Arguments
- `ensemble_path` : directory containing gauge configuration files
- `lma_path`      : directory containing per-config LMA subdirectories
- `lp`            : `SpaceParm` describing the lattice geometry
- `output_path`   : path to the HDF5 file to create (overwritten if it exists)
- `gamma`         : gamma structure for LMA, default `"g5-g5"`
- `em`            : eigenmode set `"PA"` (32 modes) or `"VV"` (64 modes)
- `config_fmt`    : gauge reader format, default `"cern"`
- `save_matrix`   : if `true`, also write `plaq_matrix` (untraced SU3, `ComplexF64[6,bsz,npls,rsz]`).
                    Default `false` — the matrix representation is large (~8× the scalar)
                    and only needed for the Phase 2 equivariant architecture.
- `verbose`       : print progress for each configuration

# HDF5 layout
See [Dataset.jl] source for the full schema. `plaq_scalar` is always written.
`plaq_matrix` is written only when `save_matrix=true`.
"""
function build_dataset(ensemble_path::String,
                       lma_path::String,
                       lp,
                       output_path::String;
                       gamma::String       = "g5-g5",
                       em::String          = "VV",
                       config_fmt::String  = "cern",
                       save_matrix::Bool   = false,
                       verbose::Bool       = true)

    if !isdir(ensemble_path)
        error("Gauge config directory not found: $(ensemble_path)")
    end
    if !isdir(lma_path)
        error("LMA directory not found: $(lma_path)")
    end

    # Gauge configs: files whose name contains a trailing n<integer>, e.g. "A654r000n1"
    # Build a map: config_id (String) → filename
    gauge_map = Dict{String,String}()
    for f in readdir(ensemble_path)
        isdir(joinpath(ensemble_path, f)) && continue
        id = _config_id(f)
        id === nothing || (gauge_map[id] = f)
    end

    # LMA configs: sub-directories under lma_path whose name is a plain integer
    lma_ids = sort(filter(f -> isdir(joinpath(lma_path, f)) && _is_integer(f),
                          readdir(lma_path)),
                   by = x -> parse(Int, x))

    common = intersect(Set(keys(gauge_map)), Set(lma_ids))

    missing_lma   = setdiff(Set(keys(gauge_map)), common)
    missing_gauge = setdiff(Set(lma_ids), common)
    isempty(missing_lma)   || @warn "$(length(missing_lma)) gauge config(s) have no LMA data: $(sort(collect(missing_lma)))"
    isempty(missing_gauge) || @warn "$(length(missing_gauge)) LMA dir(s) have no gauge config: $(sort(collect(missing_gauge)))"

    cfg_ids = sort(collect(common), by = x -> parse(Int, x))
    if isempty(cfg_ids)
        error("No matching gauge+LMA configuration pairs found.\n" *
              "  gauge dir : $(ensemble_path)\n" *
              "  LMA dir   : $(lma_path)\n" *
              "Gauge filenames must contain a trailing n<integer> (e.g. A654r000n1).")
    end

    verbose && @info "Building dataset: $(length(cfg_ids)) configurations → $(output_path)"

    reader = _make_reader(config_fmt, lp)

    h5open(output_path, "w") do fid

        # ---- metadata ----
        meta = create_group(fid, "metadata")
        write(meta, "vol",      collect(Int64, lp.iL))
        write(meta, "svol",     collect(Int64, lp.blk))
        write(meta, "gamma",    gamma)
        write(meta, "em",       em)
        write(meta, "ensemble", ensemble_path)
        write(meta, "lma_path",    lma_path)
        write(meta, "save_matrix", Int64(save_matrix))

        cfgs_grp = create_group(fid, "configs")

        for (i, cid) in enumerate(cfg_ids)
            verbose && @info "  [$i/$(length(cfg_ids))] config $(cid)"

            # --- gauge features ---
            gauge_file  = joinpath(ensemble_path, gauge_map[cid])
            U           = reader(gauge_file)
            plaq_scalar = plaquette_scalar_field(U, lp)    # Float64[bsz, npls, rsz]

            # --- correlator target (re only) ---
            lma_dir    = joinpath(lma_path, cid)
            lcfg       = get_LMAConfig_all_sources(lma_dir, gamma; em=em, re_only=true)
            re_dict    = lcfg.data["re"]                   # OrderedDict src → Vector{Float64}

            sources    = collect(keys(re_dict))            # String[nsrcs]
            T          = length(re_dict[sources[1]])
            nsrcs      = length(sources)
            correlator = Matrix{Float64}(undef, T, nsrcs)  # Float64[T, nsrcs]
            for (j, src) in enumerate(sources)
                correlator[:, j] = re_dict[src]
            end

            # --- write to HDF5 ---
            grp = create_group(cfgs_grp, cid)
            write(grp, "plaq_scalar", plaq_scalar)
            if save_matrix
                plaq_matrix = _su3_to_array(plaquette_field(U, lp))
                write(grp, "plaq_matrix", plaq_matrix)
            end
            write(grp, "correlator",  correlator)
            write(grp, "sources",     sources)
        end
    end

    verbose && @info "Dataset written to $(output_path)"
    return output_path
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_is_integer(s::String) = !isnothing(tryparse(Int, s))

"""
    _config_id(filename) -> String or nothing

Extract the config id from a gauge filename with convention `<prefix>n<integer>`.
Returns the integer part as a String, or `nothing` if the pattern is not found.

Examples: `"A654r000n1"` → `"1"`, `"A654r000n42"` → `"42"`.
"""
function _config_id(filename::String)
    m = match(r"n(\d+)$", filename)
    m === nothing ? nothing : m.captures[1]
end

function _make_reader(fmt::String, lp)
    if fmt == "cern"
        return path -> import_cern64(path, 0, lp; log=false)
    else
        error("Unsupported config format \"$(fmt)\". Only \"cern\" is currently supported.")
    end
end

"""
    _su3_to_array(plaq::Array{SU3{T}, 3}) -> Array{Complex{T}, 4}

Convert `(bsz, npls, rsz)` array of `SU3{T}` elements to a
`ComplexF64[6, bsz, npls, rsz]` array by extracting the 6 complex entries
of the first two rows: `[u11, u12, u13, u21, u22, u23]`.

The third row is not stored — it is fully determined by SU(3) unitarity:
`u3i = conj(u1j * u2k - u1k * u2j)`.
"""
function _su3_to_array(plaq::Array{SU3{T}, 3}) where {T}
    bsz, npls, rsz = size(plaq)
    out = Array{Complex{T}, 4}(undef, 6, bsz, npls, rsz)
    @inbounds for r in 1:rsz, ipl in 1:npls, b in 1:bsz
        p = plaq[b, ipl, r]
        out[1, b, ipl, r] = p.u11
        out[2, b, ipl, r] = p.u12
        out[3, b, ipl, r] = p.u13
        out[4, b, ipl, r] = p.u21
        out[5, b, ipl, r] = p.u22
        out[6, b, ipl, r] = p.u23
    end
    return out
end

end # module Dataset
