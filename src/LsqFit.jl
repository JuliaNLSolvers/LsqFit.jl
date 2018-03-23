module LsqFit

    export curve_fit,
           estimate_errors,
           standard_errors,
           margin_errors,
           estimate_covar

    using Calculus
    using Distributions
    using Compat
    using OptimBase

    import Base.summary

    include("levenberg_marquardt.jl")
    include("curve_fit.jl")

end
