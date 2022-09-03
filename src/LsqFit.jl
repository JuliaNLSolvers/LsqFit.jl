module LsqFit

export curve_fit,
    margin_error,
    confidence_interval,
    estimate_covar,
    make_hessian,
    Avv,
    # StatsBase reexports
    dof,
    coef,
    nobs,
    mse,
    rss,
    stderror,
    weights,
    residuals

using Distributions
using LinearAlgebra
using ForwardDiff
using Printf

import NLSolversBase:
    value, value!, jacobian, jacobian!, value_jacobian!!, OnceDifferentiable
import StatsBase
import StatsBase: coef, dof, nobs, rss, stderror, weights, residuals

import Base.summary

include("geodesic.jl")
include("levenberg_marquardt.jl")
include("curve_fit.jl")

end
