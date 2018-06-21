using Documenter, LsqFit

makedocs(
    format = :html,
    sitename = "LsqFit.jl",
    doctest = false,
    strict = false,
    pages = Any[
            "Home" => "index.md",
            "Getting Started" => "getting_started.md",
            "Tutorial" => "tutorial.md",
            "API References" => "api.md",
            ],
    # Use clean URLs, unless built as a "local" build
    html_prettyurls = !("local" in ARGS),
    html_canonical = "https://julianlsolvers.github.io/LineSearches.jl/stable/"
    )

deploydocs(
    repo = "github.com/JuliaNLSolvers/LsqFit.jl.git",
    target = "build",
    julia = "0.6",
    deps = nothing,
    make = nothing,
)
