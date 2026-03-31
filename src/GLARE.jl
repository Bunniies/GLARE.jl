module GLARE

include("IO.jl")
using .IO
export import_cern64, set_reader

include("Plaquette.jl")
using .Plaquette
export plaquette_field, plaquette_scalar_field

include("Correlator.jl")
using .Correlator
export LMAConfig, read_contrib_all_sources, get_LMAConfig_all_sources

include("Dataset.jl")
using .Dataset
export build_dataset

include("Preprocessing.jl")
using .Preprocessing
export split_configs, NormStats, compute_normalization, save_normalization,
       load_normalization, load_config, load_split


end # module GLARE
