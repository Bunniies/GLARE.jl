import Pkg

# Activate the docs project so Documenter is available,
# then also add the main GLARE project to LOAD_PATH so that
# GLARE and all its deps (resolved in the main Manifest) are visible.
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
        prettyurls       = get(ENV, "CI", nothing) == "true",
        canonical        = "https://Bunniies.github.io/GLARE.jl",
        edit_link        = "main",
        assets           = String[],
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
)
