using Documenter, LsqFit

makedocs(
    format = Documenter.HTML(prettyurls = true, canonical="https://julianlsolvers.github.io/LsqFit.jl/stable/"),
    sitename = "LsqFit.jl",
    doctest = false,
    warnonly = true,
    pages = Any[
            "Home" => "index.md",
            "Getting Started" => "getting_started.md",
            "Tutorial" => "tutorial.md",
            "API References" => "api.md",
            ],
    )

deploydocs(
    repo = "github.com/JuliaNLSolvers/LsqFit.jl.git",
    target = "build",
    versions = ["stable" => "v^", "v#.#"],
    deps = nothing,
    make = nothing,
)
