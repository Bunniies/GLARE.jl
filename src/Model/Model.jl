module Model

using Flux
using NNlib
using Statistics
using Zygote
using TimerOutputs
import Flux.Functors

import ..GLARE_TIMER

export PeriodicConv4D, build_baseline_cnn, pearson_r, pearson_r_loss   # Phase 1
export su3_reconstruct, plaquette_matrices                                              # Phase 2 — inputs
export BilinearLayer, ScalarGate, TracePool, GaugeEquivConv, LCBBlock                  # Phase 2 — layers
export LCNN, build_lcnn, profile_forward                                                 # Phase 2 — full model

include("BaselineCNN.jl")
include("LCNN.jl")

end # module Model
