module LsqFit

export curve_fit,
    margin_error,
    make_hessian,
    Avv,
    # StatsAPI reexports
    dof,
    coef,
    confint,
    nobs,
    mse,
    rss,
    stderror,
    weights,
    residuals,
    vcov

using Distributions
using LinearAlgebra
using ForwardDiff
using Printf
using StatsAPI

import NLSolversBase:
    value, value!, jacobian, jacobian!, value_jacobian!!, OnceDifferentiable
using StatsAPI: coef, confint, dof, nobs, rss, stderror, weights, residuals, vcov

import Base.summary

# 
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require Measurements="eff96d63-e80a-5855-80a2-b1b0885c5ab7" begin
             include("../ext/MeasurementsExt.jl")
         end
    end
end


include("geodesic.jl")
include("levenberg_marquardt.jl")
include("curve_fit.jl")

end
