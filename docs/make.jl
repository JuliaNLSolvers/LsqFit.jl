using Documenter, LsqFit

makedocs(
    format = :html,
    sitename = "LsqFit.jl",
    doctest = false,
    strict = false,
    pages = Any[
            "Home" => "index.md",
            "Getting Started" => "getting_started.md",
            # "Examples" => GENERATEDEXAMPLES,
            "API" => "api.md",
            ],
    # Use clean URLs, unless built as a "local" build
    html_prettyurls = !("local" in ARGS),
    html_canonical = "https://juliadocs.github.io/Documenter.jl/stable/",
    )

deploydocs(
    repo = "github.com/JuliaNLSolvers/LsqFit.jl.git",
    target = "build",
    julia = "0.6",
    deps = nothing,
    make = nothing,
)
