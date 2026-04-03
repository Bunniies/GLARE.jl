#!/usr/bin/env julia
# Run this locally to build and push docs to gh-pages:
#   julia docs/deploy.jl

import Pkg
Pkg.activate(@__DIR__)
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter
using Documenter: Remotes
using GLARE

makedocs(;
    modules  = [GLARE],
    sitename = "GLARE.jl",
    authors  = "Alessandro Conigli",
    repo     = Remotes.GitHub("Bunniies", "GLARE.jl"),
    format   = Documenter.HTML(;
        prettyurls = true,
        canonical  = "https://Bunniies.github.io/GLARE.jl",
        edit_link  = "main",
        assets     = String[],
    ),
    checkdocs = :exports,
    pages = [
        "Home"      => "index.md",
        "API"       => [
            "IO & Plaquette"  => "api/io.md",
            "Correlator"      => "api/correlator.md",
            "Dataset"         => "api/dataset.md",
            "Preprocessing"   => "api/preprocessing.md",
            "Model"           => "api/model.md",
        ],
    ],
)

deploydocs(;
    repo      = "github.com/Bunniies/GLARE.jl",
    target    = "build",
    branch    = "gh-pages",
    devbranch = "main",
    push_preview = false,
)
