module Preprocessing

using HDF5
using Statistics

export split_configs, NormStats, compute_normalization, compute_corr_normalization,
       save_normalization, load_normalization,
       load_gauge, load_corr, load_config, load_split,
       load_links, PreloadedDataset, preload_dataset

include("Normalization.jl")
include("DataLoading.jl")

end # module Preprocessing
