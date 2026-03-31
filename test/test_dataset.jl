using Test
using HDF5
using LatticeGPU
using GLARE

@testset "Dataset: build_dataset" begin

    # GLARE_TEST_CONF = path to a single gauge config file → dirname gives the config dir
    # GLARE_TEST_CORR = path to the LMA root dir (already the parent of per-config subdirs)
    conf_file     = joinpath(get(ENV, "GLARE_TEST_CONF", ""), "A654r000n1") # this variable is actually not used (i am reading all the available configs and not just 1)
    lma_path      = get(ENV, "GLARE_TEST_CORR", "")
    ensemble_path = isempty(conf_file) ? "" : dirname(conf_file)
    # println("\n", conf_file)
    # println(lma_path)
    # println(ensemble_path, "\n")

    if isempty(ensemble_path) || isempty(lma_path) ||
       !isdir(ensemble_path)  || !isdir(lma_path)
        @warn "Skipping dataset test: set ENV[\"GLARE_TEST_CONF\"] (path to one gauge config file) " *
              "and ENV[\"GLARE_TEST_CORR\"] (LMA root dir containing per-config subdirs)."
        return
    end

    VOL   = (48, 24, 24, 24)
    SVOL  = (8, 4, 4, 4)
    BC    = BC_PERIODIC
    TWIST = (0, 0, 0, 0, 0, 0)
    lp    = SpaceParm{4}(VOL, SVOL, BC, TWIST)

    output_path = tempname() * ".h5"

    build_dataset(ensemble_path, lma_path, lp, output_path;
                  gamma = "g5-g5", em = "VV", save_matrix = true, verbose = true)

    @test isfile(output_path)

    h5open(output_path, "r") do fid

        @test haskey(fid, "metadata")
        @test haskey(fid, "configs")

        meta = fid["metadata"]
        @test read(meta["gamma"]) == "g5-g5"
        @test read(meta["em"])    == "VV"

        cfg_ids = keys(fid["configs"])
        @test length(cfg_ids) >= 1

        cid = first(cfg_ids)
        grp = fid["configs"][cid]

        # traced scalar features
        plaq_scalar = read(grp["plaq_scalar"])
        @test plaq_scalar isa Array{Float64, 3}
        @test size(plaq_scalar) == (lp.bsz, lp.npls, lp.rsz)
        @test all(-3.0 .<= plaq_scalar .<= 3.0)

        # untraced matrix features
        plaq_matrix = read(grp["plaq_matrix"])
        @test ndims(plaq_matrix) == 4
        @test size(plaq_matrix) == (6, lp.bsz, lp.npls, lp.rsz)

        # correlator
        correlator = read(grp["correlator"])
        sources    = read(grp["sources"])
        @test correlator isa Matrix{Float64}
        T, nsrcs = size(correlator)
        @test T > 0
        @test nsrcs == length(sources)
        @test nsrcs > 0

        @info "Dataset: config=$(cid), T=$(T), nsrcs=$(nsrcs), ncfgs=$(length(cfg_ids))"
    end

    rm(output_path)

end
