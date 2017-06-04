module LsqFit

    export curve_fit,
           estimate_errors,
           estimate_covar,
           linfit

    using Calculus
    using Distributions
    using Compat
    using OptimBase

    import Base.summary

    include("levenberg_marquardt.jl")
    include("curve_fit.jl")

end
