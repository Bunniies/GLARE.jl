module Correlator

using DelimitedFiles, OrderedCollections

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

"""Supported gamma structures for LMA correlator files."""
const GAMMA = ["g5-g5", "g5-g0g5", "g1-g1", "g2-g2", "g3-g3"]

"""LMA contribution labels."""
const CONTRIB = ["re", "rr", "ee"]

# ---------------------------------------------------------------------------
# LMAConfig — holds all LMA contributions for one gauge configuration
# ---------------------------------------------------------------------------

"""
    LMAConfig

Stores the LMA two-point function contributions for a single gauge configuration.

Fields:
- `ncnfg`    : configuration number (parsed from directory name)
- `gamma`    : gamma structure used (e.g. `"g5-g5"`)
- `eigmodes` : number of low eigenmodes used (32 for PA, 64 for VV)
- `data`     : `Dict` with keys `"ee"`, `"re"`, `"rr"`, each an
               `OrderedDict{String, Vector{Float64}}` keyed by source position.

The rest-eigen correlator at source `"0"` is `data["re"]["0"]`.
"""
mutable struct LMAConfig
    ncnfg    :: Int64
    gamma    :: String
    eigmodes :: Int64
    data     :: Dict{Any, Any}

    LMAConfig(ncnfg, gamma, eigmodes, data) = new(ncnfg, gamma, eigmodes, data)
end

function Base.show(io::IO, a::LMAConfig)
    println(io, "LMAConfig")
    println(io, " - Ncnfg     : ", a.ncnfg)
    println(io, " - Gamma     : ", a.gamma)
    println(io, " - Eigenmodes: ", a.eigmodes)
end

export LMAConfig

# ---------------------------------------------------------------------------
# read_contrib_all_sources
# ---------------------------------------------------------------------------

@doc raw"""
    read_contrib_all_sources(path::String, g::String) -> OrderedDict{String, Vector{Float64}}

Read one LMA contribution file (ee, re, or rr) for a **single configuration**
that stores **all source positions** in the same file (files with extension
`mseig**ee.dat`, `mseig**re.dat`, `mseig**rr.dat`).

Returns an `OrderedDict` whose keys are the source positions (as strings, e.g.
`"0"`, `"12"`, ...) and whose values are `Vector{Float64}` of length `T`
(number of time slices).

# Arguments
- `path`  : full path to the `.dat` file
- `g`     : gamma structure, must be one of `GAMMA` (e.g. `"g5-g5"`)
"""
function read_contrib_all_sources(path::String, g::String)
    if !(g in GAMMA)
        error("Gamma structure \"$(g)\" is not supported. " *
              "Valid choices: $(GAMMA)")
    end

    f = readdlm(path)

    dlm_tsrc = findall(x -> typeof(x) <: AbstractString && occursin("#tsrc", x), f)
    dlm_g    = findall(x -> typeof(x) <: AbstractString && occursin("#" * g, x), f)

    # Determine tvals dynamically: count data rows between first and second
    # comment line after the first gamma delimiter.
    idx_time = findall(x -> typeof(x) <: AbstractString && occursin("#", x),
                       f[dlm_g[1]:dlm_g[2]])
    tvals = length(f[idx_time[1].I[1]+1 : idx_time[2].I[1]-1, 1])

    if length(dlm_tsrc) != length(dlm_g)
        error("Found $(length(dlm_tsrc)) source delimiters and " *
              "$(length(dlm_g)) gamma delimiters in $(path).\n" *
              "Check that gamma \"$(g)\" is present for every source position.")
    end

    datadict = OrderedDict{String, Vector{Float64}}()
    for k in eachindex(dlm_tsrc)
        tsrc = split(f[dlm_tsrc[k]], "=")[end]
        idx  = dlm_g[k].I[1]
        datadict[tsrc] = Float64.(f[idx+1 : idx+tvals, 2])
    end

    return datadict
end

export read_contrib_all_sources

# ---------------------------------------------------------------------------
# get_LMAConfig_all_sources
# ---------------------------------------------------------------------------

@doc raw"""
    get_LMAConfig_all_sources(path::String, g::String;
                              em::String="VV",
                              bc::Bool=false,
                              re_only::Bool=false) -> LMAConfig

Read the LMA contributions (ee, re, rr) for a single configuration stored in
`path`, where every contribution has all source positions in one file.

Expected files in `path` (e.g. `path = ".../r000n1/0001234"`):

    mseig<em>ee.dat    — eigen-eigen contribution
    mseig<em>re.dat    — rest-eigen contribution
    mseig<em>rr.dat    — rest-rest contribution

# Arguments
- `path`    : directory for one configuration (its `basename` is parsed as `ncnfg`)
- `g`       : gamma structure, must be one of `GAMMA`
- `em`      : eigenmode set: `"PA"` (32 modes) or `"VV"` (64 modes)
- `bc`      : if `true`, attempt to read bias-correction data (not yet implemented)
- `re_only` : if `true`, read only the rest-eigen contribution (skip ee and rr)

# Returns
An `LMAConfig` with:
- `data["ee"]` : `OrderedDict{String, Vector{Float64}}` keyed by source position
- `data["re"]` : same
- `data["rr"]` : same (omitted when `re_only=true`)
"""
function get_LMAConfig_all_sources(path::String, g::String;
                                   em::String="VV",
                                   bc::Bool=false,
                                   re_only::Bool=false)

    modes = Dict("PA" => 32, "VV" => 64)
    if !haskey(modes, em)
        error("Unknown eigenmode set \"$(em)\". Valid choices: $(collect(keys(modes)))")
    end

    f = readdir(path)

    p_ee = filter(x -> occursin(string("mseig", em, "ee"), x), f)
    p_re = filter(x -> occursin(string("mseig", em, "re"), x), f)
    p_rr = filter(x -> occursin(string("mseig", em, "rr"), x), f)

    if length(p_re) == 0
        error("No rest-eigen file (mseig$(em)re.dat) found in $(path)")
    end
    if !re_only && (length(p_ee) == 0 || length(p_rr) == 0)
        error("Missing ee or rr files in $(path).\n" *
              "Set re_only=true to read only the rest-eigen contribution.")
    end

    if bc
        error("Reading bias-correction data is not yet implemented.")
    end

    res_dict = Dict{Any,Any}()
    res_dict["re"] = read_contrib_all_sources(joinpath(path, p_re[1]), g)
    if !re_only
        res_dict["ee"] = read_contrib_all_sources(joinpath(path, p_ee[1]), g)
        res_dict["rr"] = read_contrib_all_sources(joinpath(path, p_rr[1]), g)
    end

    return LMAConfig(parse(Int64, basename(path)), g, modes[em], res_dict)
end

export get_LMAConfig_all_sources

end # module Correlator
