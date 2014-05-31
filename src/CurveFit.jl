module CurveFit
    export curve_fit, estimate_errors, levenberg_marquardt

    using Calculus
    using Optim
    using Distributions
    
    include("levenberg_marquardt.jl")
    include("curve_fit.jl")

end
