using GLARE
using LatticeGPU
using HDF5

# ---------------------------------------------------------------------------
# Lattice parameters for CLS ensemble A654
# ---------------------------------------------------------------------------
VOL   = (48, 24, 24, 24)
SVOL  = (8, 4, 4, 4)
BC    = BC_PERIODIC
TWIST = (0, 0, 0, 0, 0, 0)
lp    = SpaceParm{4}(VOL, SVOL, BC, TWIST)

ensemble_path = "/Users/alessandroconigli/Lattice/data/cls"
lma_path      = "/Users/alessandroconigli/Lattice/data/HVP/LMA/A654_all_t_sources/dat"

POLARIZATIONS = ["g1-g1", "g2-g2", "g3-g3"]
EM            = "VV"

outdir     = "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources"
scalar_h5  = joinpath(outdir, "A654_gauge_scalar.h5")
matrix_h5  = joinpath(outdir, "A654_gauge_matrix.h5")
corr_h5    = joinpath(outdir, "A654_corr.h5")

# ---------------------------------------------------------------------------
# Build scalar gauge database (all configs)
# ---------------------------------------------------------------------------
build_gauge_dataset(ensemble_path, lp, scalar_h5; verbose=true)

# ---------------------------------------------------------------------------
# Build matrix gauge database (all configs)
# Separate file — only needed for Phase 2 (L-CNN). Can be skipped for now.
# ---------------------------------------------------------------------------
# build_gauge_matrix_dataset(ensemble_path, lp, matrix_h5; verbose=true)

# ---------------------------------------------------------------------------
# Build correlator database (all three vector polarizations)
# ---------------------------------------------------------------------------
build_corr_dataset(lma_path, corr_h5;
                   em=EM, polarizations=POLARIZATIONS, verbose=true)

# ---------------------------------------------------------------------------
# Inspect the resulting files
# ---------------------------------------------------------------------------
println("\n--- Scalar gauge database ---")
h5open(scalar_h5, "r") do fid
    cfg_ids = sort(keys(fid["configs"]), by=x->parse(Int,x))
    println("configs : $(length(cfg_ids))")

    cid = first(cfg_ids)
    ps  = read(fid["configs"][cid]["plaq_scalar"])
    println("plaq_scalar shape : $(size(ps))  (iL×npls = $(lp.iL)×$(lp.npls))")
    println("plaq_scalar range : [$(minimum(ps)), $(maximum(ps))]")
end

println("\n--- Correlator database ---")
h5open(corr_h5, "r") do fid
    cfg_ids = sort(keys(fid["configs"]), by=x->parse(Int,x))
    println("configs        : $(length(cfg_ids))")
    println("em             : $(read(fid["metadata"]["em"]))")
    println("polarizations  : $(read(fid["metadata"]["polarizations"]))")

    cid = first(cfg_ids)
    for pol in POLARIZATIONS
        co  = read(fid["configs"][cid][pol]["correlator"])
        src = read(fid["configs"][cid][pol]["sources"])
        println("  $(pol): correlator $(size(co)), $(length(src)) sources")
    end
end
