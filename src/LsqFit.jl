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

import PrecompileTools: @compile_workload

@compile_workload begin
    model(x, p) = @. p[1] * x + p[2]

    local fit
    for T in (Float32, Float64)
        xdata = T.(1:100)
        ydata = model(xdata, T[1.0, 1.0])
        p0 = T[0.8, 0.8]

        for ad in (:finite, :finiteforward, :forwarddiff)
            fit = curve_fit(model, xdata, ydata, p0; autodiff=ad)
        end
    end

    stderror(fit)
    margin_error(fit)
    confint(fit)
    vcov(fit)
end

end
