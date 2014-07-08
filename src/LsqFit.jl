module LsqFit

    export curve_fit,
           estimate_errors,
           estimate_covar,
           linfit

    using Optim
    using Calculus
    using Distributions

    include("curve_fit.jl")

end
