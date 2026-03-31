using Test
using Statistics
using LatticeGPU
using GLARE

@testset "Plaquette: plaquette_field and plaquette_scalar_field" begin

    # ---------------------------------------------------------------------------
    # Lattice parameters (CLS A654)
    # ---------------------------------------------------------------------------
    VOL   = (48, 24, 24, 24)
    SVOL  = (8, 4, 4, 4)
    BC    = BC_PERIODIC
    TWIST = (0, 0, 0, 0, 0, 0)
    lp    = SpaceParm{4}(VOL, SVOL, BC, TWIST)

    path_config = joinpath(get(ENV, "GLARE_TEST_CONF", ""), "A654r000n1")

    if !isfile(path_config)
        @warn "Skipping plaquette test: config file not found at $path_config."
        return
    end

    U = set_reader("cern", lp)(path_config)

    # ---------------------------------------------------------------------------
    # plaquette_scalar_field: shape
    # ---------------------------------------------------------------------------
    scalars = plaquette_scalar_field(U, lp)

    @test scalars isa Array{Float64, 3}
    @test size(scalars) == (lp.bsz, lp.npls, lp.rsz)

    # ---------------------------------------------------------------------------
    # plaquette_scalar_field: global average must be in physical range.
    # The CERN header stores Re(Tr P) averaged ≈ 1.57 for this ensemble.
    # Our avg_plaq ≈ header/3 ≈ 0.51, consistent with the standard plaquette
    # observable Re(Tr P)/Nc ≈ 0.52 for CLS A654 (β ≈ 3.41).
    # ---------------------------------------------------------------------------
    avg_plaq = sum(scalars) / (prod(lp.iL) * lp.npls)
    @test 0.1 < avg_plaq < 3.0

    @info "Average plaquette = $(round(avg_plaq, digits=6))  " *
          "(CERN header ≈ $(round(avg_plaq * 3, digits=4)))"

    # ---------------------------------------------------------------------------
    # plaquette_scalar_field: all values in [-3, 3] (Re(Tr P) bound for SU3)
    # ---------------------------------------------------------------------------
    @test all(-3.0 .<= scalars .<= 3.0)

    # ---------------------------------------------------------------------------
    # plaquette_field: shape and element type
    # ---------------------------------------------------------------------------
    T = Float64
    plaq = plaquette_field(U, lp)

    @test plaq isa Array{SU3{T}, 3}
    @test size(plaq) == (lp.bsz, lp.npls, lp.rsz)

    # ---------------------------------------------------------------------------
    # plaquette_field: each P_μν(x) must be SU(3), i.e. P · P† = I.
    # dev_one measures distance from the *identity* matrix, so the correct
    # SU(3) closure test is dev_one(P / P) where P/P = P · P† (since `/` is
    # right-multiply by conjugate transpose in LatticeGPU). This should vanish.
    # ---------------------------------------------------------------------------
    max_dev = maximum(dev_one(plaq[b, ipl, r] / plaq[b, ipl, r])
                      for b in 1:lp.bsz, ipl in 1:lp.npls, r in 1:lp.rsz)
    @test max_dev < 1e-10

    # ---------------------------------------------------------------------------
    # Consistency: plaquette_scalar_field == Re(Tr(plaquette_field))
    # ---------------------------------------------------------------------------
    scalar_from_plaq = real.(tr.(plaq))
    @test scalar_from_plaq ≈ scalars

    # ---------------------------------------------------------------------------
    # Plane ordering: lp.plidx[ipl] = (id1, id2) matches the loop order in
    # plaquette_field (id1: N→1, id2: 1→id1-1).
    # Spot-check plane 1 manually at site (b=1, r=1).
    # ---------------------------------------------------------------------------
    b, r = 1, 1
    id1, id2 = lp.plidx[1]   # should be (N, N-1) = (4, 3) for 4D
    bu1, ru1 = up((b, r), id1, lp)
    bu2, ru2 = up((b, r), id2, lp)
    P_manual = U[b, id1, r] * U[bu1, id2, ru1] / (U[b, id2, r] * U[bu2, id1, ru2])
    @test dev_one(plaq[b, 1, r] / P_manual) < 1e-12

end
