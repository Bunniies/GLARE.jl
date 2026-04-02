using Test
using HDF5
using LatticeGPU
using GLARE

@testset "Dataset: build_gauge_dataset + build_gauge_matrix_dataset + build_corr_dataset" begin

    # GLARE_TEST_CONF  = path to a single gauge config file; dirname → gauge dir
    # GLARE_TEST_CORR  = LMA root dir containing per-config integer subdirs
    conf_file     = get(ENV, "GLARE_TEST_CONF", "")
    lma_path      = get(ENV, "GLARE_TEST_CORR", "")
    ensemble_path = isempty(conf_file) ? "" : dirname(conf_file)

    if isempty(ensemble_path) || isempty(lma_path) ||
       !isdir(ensemble_path)  || !isdir(lma_path)
        @warn "Skipping dataset test: set ENV[\"GLARE_TEST_CONF\"] (path to a gauge config file) " *
              "and ENV[\"GLARE_TEST_CORR\"] (LMA root dir with per-config integer subdirs)."
        return
    end

    VOL   = (48, 24, 24, 24)
    SVOL  = (8, 4, 4, 4)
    BC    = BC_PERIODIC
    TWIST = (0, 0, 0, 0, 0, 0)
    lp    = SpaceParm{4}(VOL, SVOL, BC, TWIST)

    POLARIZATIONS = ["g1-g1", "g2-g2", "g3-g3"]
    EM            = "VV"

    scalar_h5 = tempname() * "_gauge_scalar.h5"
    matrix_h5 = tempname() * "_gauge_matrix.h5"
    corr_h5   = tempname() * "_corr.h5"

    # -----------------------------------------------------------------------
    # build_gauge_dataset (scalar only, config_range)
    # -----------------------------------------------------------------------
    build_gauge_dataset(ensemble_path, lp, scalar_h5;
                        config_range=1:1, verbose=true)
    @test isfile(scalar_h5)

    h5open(scalar_h5, "r") do fid
        @test haskey(fid, "metadata")
        @test haskey(fid, "configs")

        meta = fid["metadata"]
        @test !haskey(meta, "save_matrix")   # removed from scalar db

        cfg_ids = keys(fid["configs"])
        @test length(cfg_ids) == 1

        cid = first(cfg_ids)
        grp = fid["configs"][cid]

        @test haskey(grp, "plaq_scalar")
        @test !haskey(grp, "plaq_matrix")

        ps = read(grp["plaq_scalar"])
        @test ps isa Array{Float64, 5}
        @test size(ps) == (lp.iL[1], lp.iL[2], lp.iL[3], lp.iL[4], lp.npls)
        @test all(-3.0 .<= ps .<= 3.0)

        @info "Scalar gauge dataset: $(length(cfg_ids)) config, shape=$(size(ps))"
    end

    # -----------------------------------------------------------------------
    # build_gauge_matrix_dataset (matrix only, config_range)
    # -----------------------------------------------------------------------
    build_gauge_matrix_dataset(ensemble_path, lp, matrix_h5;
                                config_range=1:1, verbose=true)
    @test isfile(matrix_h5)

    h5open(matrix_h5, "r") do fid
        @test haskey(fid, "metadata")
        @test haskey(fid, "configs")

        cfg_ids = keys(fid["configs"])
        @test length(cfg_ids) == 1

        cid = first(cfg_ids)
        grp = fid["configs"][cid]

        @test haskey(grp, "plaq_matrix")
        @test !haskey(grp, "plaq_scalar")

        pm = read(grp["plaq_matrix"])
        @test pm isa Array{<:Complex, 6}
        @test size(pm) == (6, lp.iL[1], lp.iL[2], lp.iL[3], lp.iL[4], lp.npls)

        @info "Matrix gauge dataset: $(length(cfg_ids)) config, shape=$(size(pm))"
    end

    # -----------------------------------------------------------------------
    # build_corr_dataset (config_range)
    # -----------------------------------------------------------------------
    build_corr_dataset(lma_path, corr_h5;
                       em=EM, polarizations=POLARIZATIONS,
                       config_range=1:1, verbose=true)
    @test isfile(corr_h5)

    h5open(corr_h5, "r") do fid
        @test haskey(fid, "metadata")
        @test haskey(fid, "configs")

        meta = fid["metadata"]
        @test read(meta["em"]) == EM
        @test read(meta["polarizations"]) == POLARIZATIONS

        cfg_ids = keys(fid["configs"])
        @test length(cfg_ids) == 1

        cid = first(cfg_ids)
        grp = fid["configs"][cid]

        for pol in POLARIZATIONS
            @test haskey(grp, pol)
            correlator = read(grp[pol]["correlator"])
            sources    = read(grp[pol]["sources"])
            @test correlator isa Matrix{Float64}
            T, nsrcs = size(correlator)
            @test T > 0
            @test nsrcs == length(sources)
            @test nsrcs > 0
        end

        @info "Correlator dataset: $(length(cfg_ids)) config, polarizations=$(POLARIZATIONS)"
    end

    rm(scalar_h5)
    rm(matrix_h5)
    rm(corr_h5)
end
