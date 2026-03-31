using Test
using GLARE
using OrderedCollections

@testset "Correlator: LMA rest-eigen reader" begin

    # ---------------------------------------------------------------------------
    # Path to a single-configuration LMA data directory.
    # Expected layout:
    #   <path>/
    #     mseig<em>ee.dat
    #     mseig<em>re.dat
    #     mseig<em>rr.dat
    # Override via ENV["GLARE_TEST_CORR"].
    # ---------------------------------------------------------------------------
    ENV["GLARE_TEST_CORR"] = "/Users/alessandroconigli/Lattice/data/HVP/LMA/A654_all_t_sources/dat"
    path_corr = joinpath(get(ENV, "GLARE_TEST_CORR", ""), "1")

    if isempty(path_corr) || !isdir(path_corr)
        @warn "Skipping correlator test: set ENV[\"GLARE_TEST_CORR\"] to a " *
              "directory containing mseigVVee/re/rr.dat files."
        return
    end

    GAMMA_TEST = "g5-g5"
    EM         = "VV"

    # ---------------------------------------------------------------------------
    # read_contrib_all_sources: re contribution only
    # ---------------------------------------------------------------------------
    p_re = joinpath(path_corr, filter(x -> occursin("mseig$(EM)re", x), readdir(path_corr))[1])
    re_dict = read_contrib_all_sources(p_re, GAMMA_TEST)

    @test re_dict isa OrderedCollections.OrderedDict
    @test length(re_dict) > 0                        # at least one source

    tsrc0 = first(keys(re_dict))
    @test re_dict[tsrc0] isa Vector{Float64}
    @test length(re_dict[tsrc0]) > 0                 # non-empty time series

    # ---------------------------------------------------------------------------
    # get_LMAConfig_all_sources: full LMA config
    # ---------------------------------------------------------------------------
    cfg = get_LMAConfig_all_sources(path_corr, GAMMA_TEST; em=EM)

    @test cfg isa LMAConfig
    @test cfg.gamma    == GAMMA_TEST
    @test cfg.eigmodes == 64                         # VV → 64 modes

    for key in ("ee", "re", "rr")
        @test haskey(cfg.data, key)
        @test cfg.data[key] isa OrderedCollections.OrderedDict
        @test length(cfg.data[key]) > 0
    end

    # All contributions share the same source positions and time extent
    srcs_ee = collect(keys(cfg.data["ee"]))
    srcs_re = collect(keys(cfg.data["re"]))
    srcs_rr = collect(keys(cfg.data["rr"]))
    @test srcs_ee == srcs_re == srcs_rr

    T = length(cfg.data["re"][srcs_re[1]])
    for key in ("ee", "re", "rr"), s in srcs_re
        @test length(cfg.data[key][s]) == T
    end

    @info "LMAConfig ncnfg=$(cfg.ncnfg), γ=$(cfg.gamma), " *
          "$(length(srcs_re)) sources, T=$(T)"

    # ---------------------------------------------------------------------------
    # re_only flag
    # ---------------------------------------------------------------------------
    cfg_re = get_LMAConfig_all_sources(path_corr, GAMMA_TEST; em=EM, re_only=true)
    @test  haskey(cfg_re.data, "re")
    @test !haskey(cfg_re.data, "ee")
    @test !haskey(cfg_re.data, "rr")

    # ---------------------------------------------------------------------------
    # Error on unsupported gamma
    # ---------------------------------------------------------------------------
    @test_throws ErrorException read_contrib_all_sources(p_re, "bad-gamma")

    # ---------------------------------------------------------------------------
    # Error on unknown em
    # ---------------------------------------------------------------------------
    @test_throws ErrorException get_LMAConfig_all_sources(path_corr, GAMMA_TEST; em="XX")

end
