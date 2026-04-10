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

include("Dataset/Dataset.jl")
using .Dataset
export build_gauge_dataset, build_gauge_matrix_dataset, build_gauge_link_dataset, build_corr_dataset, merge_dataset

include("Preprocessing/Preprocessing.jl")
using .Preprocessing
export split_configs, NormStats, compute_normalization, compute_corr_normalization,
       save_normalization, load_normalization,
       load_gauge, load_corr, load_config, load_split,
       load_links, PreloadedDataset, preload_dataset

include("Model/Model.jl")
using .Model
export PeriodicConv4D, build_baseline_cnn, pearson_r   # Phase 1
export su3_reconstruct, BilinearLayer, ScalarGate, TracePool, GaugeEquivConv, LCBBlock, plaquette_matrices  # Phase 2
export LCNN, build_lcnn  # Phase 2 — full model

end # module GLARE
