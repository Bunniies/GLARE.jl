using GLARE
using LatticeGPU
using Test
using HDF5
# ---------------------------------------------------------------------------
# Lattice parameters for CLS ensemble A654
# VOL = (48, 24, 24, 24),  SVOL = (8, 4, 4, 4),  periodic BC
# ---------------------------------------------------------------------------
VOL   = (48, 24, 24, 24)
SVOL  = (8, 4, 4, 4)
BC    = BC_PERIODIC
TWIST = (0, 0, 0, 0, 0, 0)
lp    = SpaceParm{4}(VOL, SVOL, BC, TWIST)

path_to_cnfg = "/Users/alessandroconigli/Lattice/data/cls/"
path_to_corr = "/Users/alessandroconigli/Lattice/data/HVP/LMA/A654_all_t_sources/dat"
cnfgs = readdir(path_to_cnfg)

# ---------------------------------------------------------------------------
# Build a reader and load the configuration
# ---------------------------------------------------------------------------
cnfg_reader = set_reader("cern", lp)
U = cnfg_reader(joinpath(path_to_cnfg, cnfgs[1]))

# ---------------------------------------------------------------------------
# plaquette_scalar_field: shape
# ---------------------------------------------------------------------------
scalars = plaquette_scalar_field(U, lp)

# ---------------------------------------------------------------------------
# plaquette_field: shape and element type
# ---------------------------------------------------------------------------
plaq = plaquette_field(U, lp)

# ---------------------------------------------------------------------------
# read_contrib_all_sources: re contribution only
# ---------------------------------------------------------------------------
GAMMA_TEST = "g5-g5"
EM         = "VV"
fname_lma = readdir(path_to_corr)
filter!(x->x!= "convert", fname_lma)

p_re_cnfg1 = joinpath(path_to_corr, fname_lma[1])
p_re_cnfg1_redat = filter(x->occursin("mseig$(EM)re", x), readdir(p_re_cnfg1,join=true))[1]

re_dict = read_contrib_all_sources(p_re_cnfg1_redat, GAMMA_TEST)

# ---------------------------------------------------------------------------
# get_LMAConfig_all_sources: full LMA config
# ---------------------------------------------------------------------------
cnfg = get_LMAConfig_all_sources(p_re_cnfg1, GAMMA_TEST; em=EM)


# --------------------------------------------------
# buil_dataset test
# --------------------------------------------------
output_path = tempname() * ".h5"

output_path = "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_test.hf"
build_dataset(path_to_cnfg, path_to_corr, lp, output_path; gamma=GAMMA_TEST, em=EM, verbose=true)

isfile(output_path)

metadata = nothing
configsdata = nothing
h5open(output_path, "r") do fid
    global metadata = read(fid["metadata"])
    global configsdata = read(fid["configs"])
end


