module LsqFit

    export curve_fit,
           standard_error,
           margin_error,
           confidence_interval,
           estimate_covar,
           dof,
           coef

    using Distributions
    using OptimBase
    using LinearAlgebra
    import NLSolversBase: value, jacobian
    import StatsBase
    import StatsBase: dof, coef, nobs

    import Base.summary

    include("levenberg_marquardt.jl")
    include("curve_fit.jl")

end
