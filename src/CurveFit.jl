module CurveFit
    export curve_fit, estimate_errors

    using Optim
    using Calculus
    using Distributions
    
    include("curve_fit.jl")

end
