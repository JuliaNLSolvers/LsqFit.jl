module LsqFit

    export least_squares,
           curve_fit,
           estimate_covar,
           standard_error,
           margin_error,
           confidence_interval

    using NLSolversBase
    using Distributions
    using Compat

    import Base.summary

    include("types.jl")
    include("api.jl")
    include("lsqfit/least_squares.jl")
    # include("lsqfit/curve_fit.jl")
    include("solvers/levenberg_marquardt.jl")
    # include("utilities/assess_fit.jl")
end
