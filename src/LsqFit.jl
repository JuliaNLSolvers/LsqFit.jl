module LsqFit

    export curve_fit,
           estimate_errors,
           estimate_covar

    using Optim
    using Calculus
    using Distributions
    using Compat

    import Base.summary

    include("levenberg_marquardt.jl")
    include("curve_fit.jl")

end
