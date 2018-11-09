module LsqFit

    export curve_fit,
           standard_error,
           margin_error,
           confidence_interval,
           estimate_covar,
           # StatsBase reexports
           dof, coef, nobs,
           stderr, weights, residuals

    using Distributions
    using LinearAlgebra
    import NLSolversBase: value, jacobian, OnceDifferentiable
    import StatsBase
    import StatsBase: coef, dof, nobs, rss, stderr, weights, residuals

    import Base.summary

    abstract type Optimizer end
    abstract type OptimizationResults end

    struct OptimizationState{T <: Optimizer}
        iteration::Int
        value::Float64
        g_norm::Float64
        metadata::Dict
    end
    OptimizationTrace{T} = Vector{OptimizationState{T}}
    struct NonLinearLsqResults{O<:Optimizer,T,N} <: OptimizationResults
        method::O
        initial_x::Array{T,N}
        minimizer::Array{T,N}
        minimum::T
        iterations::Int
        iteration_converged::Bool
        x_converged::Bool
        x_tol::Float64
        x_residual::Float64
        f_converged::Bool
        f_tol::Float64
        f_residual::Float64
        g_converged::Bool
        g_tol::Float64
        g_residual::Float64
        f_increased::Bool
        trace::OptimizationTrace{O}
        f_calls::Int
        J_calls::Int
    end

    minimizer(nlsqr::NonLinearLsqResults) = nlsqr.minimizer
    converged(nlsqr::NonLinearLsqResults) = nlsqr.x_converged || nlsqr.f_converged || nlsqr.g_converged
    include("levenberg_marquardt.jl")
    include("curve_fit.jl")

end
