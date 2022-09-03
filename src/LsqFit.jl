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


struct LsqFitState{T}
    iteration::Int
    value::Float64
    g_norm::Float64
    metadata::Dict
end
function Base.show(io::IO, t::LsqFitState)
    @printf io "%6d   %14e   %14e\n" t.iteration t.value t.g_norm
    if !isempty(t.metadata)
        for (key, value) in t.metadata
            @printf io " * %s: %s\n" key value
        end
    end
    return
end

LsqFitTrace{T} = Vector{LsqFitState{T}}
function Base.show(io::IO, tr::LsqFitTrace)
    @printf io "Iter     Function value   Gradient norm \n"
    @printf io "------   --------------   --------------\n"
    for state in tr
        show(io, state)
    end
    return
end

struct LsqFitResults{O,T,N}
    method::O
    initial_x::Array{T,N}
    minimizer::Array{T,N}
    minimum::T
    iterations::Int
    iteration_converged::Bool
    x_converged::Bool
    g_converged::Bool
    g_tol::Float64
    trace::LsqFitTrace{O}
    f_calls::Int
    g_calls::Int
end

minimizer(lsr::LsqFitResults) = lsr.minimizer
isconverged(lsr::LsqFitResults) = lsr.x_converged || lsr.g_converged

include("geodesic.jl")
include("levenberg_marquardt.jl")
include("curve_fit.jl")

end
