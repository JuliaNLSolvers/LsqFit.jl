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

include("geodesic.jl")
include("levenberg_marquardt.jl")
include("curve_fit.jl")

end
