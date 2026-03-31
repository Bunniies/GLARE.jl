using Test
using LatticeGPU
using GLARE

@testset "IO: CLS configuration reader (CERN format)" begin

    # ---------------------------------------------------------------------------
    # Lattice parameters for CLS ensemble A654
    # VOL = (48, 24, 24, 24),  SVOL = (8, 4, 4, 4),  periodic BC
    # ---------------------------------------------------------------------------
    VOL   = (48, 24, 24, 24)
    SVOL  = (8, 4, 4, 4)
    BC    = BC_PERIODIC
    TWIST = (0, 0, 0, 0, 0, 0)
    lp    = SpaceParm{4}(VOL, SVOL, BC, TWIST)

    # Path to a CLS A654 configuration.
    # Override by setting the environment variable GLARE_TEST_CONF.
    ENV["GLARE_TEST_CONF"] ="/Users/alessandroconigli/Lattice/data/cls/"

    path_config = joinpath(get(ENV, "GLARE_TEST_CONF", ""), "A654r000n1")

    if !isfile(path_config)
        @warn "Skipping reader test: config file not found at $path_config. " *
              "Set ENV[\"GLARE_TEST_CONF\"] to point to a valid CERN-format file."
        return
    end

    # ---------------------------------------------------------------------------
    # Build a reader and load the configuration
    # ---------------------------------------------------------------------------
    cnfg_reader = set_reader("cern", lp)
    @test cnfg_reader isa Function

    U = cnfg_reader(path_config)

    # ---------------------------------------------------------------------------
    # Shape checks
    # ---------------------------------------------------------------------------
    @test U isa Array{SU3{Float64}, 3}
    @test size(U, 1) == lp.bsz    # block size
    @test size(U, 2) == lp.ndim   # 4 directions
    @test size(U, 3) == lp.rsz    # number of blocks

    
    # ---------------------------------------------------------------------------
    # set_reader: unsupported format raises an error
    # ---------------------------------------------------------------------------
    @test_throws ErrorException set_reader("hdf5", lp)

end
