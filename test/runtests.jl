using Test
using LatticeGPU
using GLARE

@testset "GLARE.jl" begin
    include("test_reader.jl")
    include("test_plaquette.jl")
    include("test_correlator.jl")
    include("test_dataset.jl")
    include("test_preprocessing.jl")
    include("test_model.jl")
end
