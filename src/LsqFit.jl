module LsqFit

    export curve_fit,
           standard_error,
           margin_error,
           confidence_interval,
           estimate_covar,
           # StatsBase reexports
           dof, coef, nobs, mse, rss,
           stderr, weights, residuals

    using Distributions
    using OptimBase
    using LinearAlgebra
    import NLSolversBase: value, jacobian
    import StatsBase
    import StatsBase: coef, dof, nobs, rss, stderr, weights, residuals

    import Base.summary

    include("levenberg_marquardt.jl")
    include("curve_fit.jl")

end
