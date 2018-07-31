immutable LevenbergMarquardt{T<:Real, V<:Vector} <: AbstractOptimizer
    min_step_quality::T
    good_step_quality::T
    initial_lambda::T
    lambda_increase::T
    lambda_decrease::T
    lower::V
    upper::V
end

Base.summary(::LevenbergMarquardt) = "Levenberg-Marquardt"

"""
    LevenbergMarquardt(; min_step_quality = 1e-3,
                         good_step_quality = 0.75,
                         initial_lambda = 10.0,
                         lambda_increase = 10.0,
                         lambda_decrease = 0.1,
                         lower = zeros(0),
                         upper = zeros(0))

Implements box constraints as described in Kanzow, Yamashita, Fukushima (2004; J
Comp & Applied Math).
"""
function LevenbergMarquardt(;min_step_quality = 1e-3,
                             good_step_quality = 0.75,
                             initial_lambda = 10.0,
                             lambda_increase = 10.0,
                             lambda_decrease = 0.1,
                             lower = zeros(0),
                             upper = zeros(0))
    LevenbergMarquardt(min_step_quality,
                       good_step_quality,
                       initial_lambda,
                       lambda_increase,
                       lambda_decrease,
                       lower,
                       upper)
end

mutable struct LevenbergMarquardtState{Tx, T, L, M, V} <: AbstractOptimizerState
    x::Tx
    previous_x::Tx
    previous_f_x::T
    lambda::L
    n_matrix::M # cache for J'J
    n_vector::V # cache for J'f_x
end

function initial_state(d, initial_x, method::LevenbergMarquardt, options)
    # check parameters
    @assert (isempty(method.lower) || length(method.lower)==length(initial_x)) && (isempty(method.upper) || length(method.upper)==length(initial_x)) "Bounds must either be empty or of the same length as the number of parameters."
    @assert (isempty(method.lower) || all(initial_x .>= method.lower)) && (isempty(method.upper) || all(initial_x .<= method.upper)) "Initial guess must be within bounds."
    @assert 0 <= method.min_step_quality < 1 "0 <= min_step_quality < 1 must hold."
    @assert 0 < method.good_step_quality <= 1 "0 < good_step_quality <= 1 must hold."
    @assert method.min_step_quality < method.good_step_quality "min_step_quality < good_step_quality must hold."

    T = eltype(initial_x)
    n = length(initial_x)
    value_gradient!!(d, initial_x)

    LevenbergMarquardtState(initial_x,
                            similar(initial_x),
                            T(NaN),
                            method.initial_lambda,
                            Matrix{T}(undef, n, n),
                            Vector{T}(undef, n))
end

function update_state!(state::LevenbergMarquardtState, d, method::LevenbergMarquardt)
    # we want to solve:
    #    argmin 0.5*||J(x)*delta_x + r(x)||^2 + lambda*||diagm(J'*J)*delta_x||^2
    # Solving for the minimum gives:
    #    (J'* J + lambda*diagm(DtD)) * delta_x == -J' * r(p), where DtD = sum(abs2, J, 1)
    # Where we have used the equivalence: diagm(J'*J) = diagm(sum(abs2, J, 1))
    # It is additionally useful to bound the elements of DtD below to help
    # prevent "parameter evaporation".

    # constants
    MAX_LAMBDA = 1e16
    MIN_LAMBDA = 1e-16
    MIN_DIAGONAL = 1e-6

    # values from objective
    J = gradient(d)
    f_x = value(d)
    residual = sum(abs2, f_x)

    DtD = vec(sum(abs2, J, 1))

    for i in 1:length(DtD)
        if DtD[i] <= MIN_DIAGONAL
            DtD[i] = MIN_DIAGONAL
        end
    end

    # delta_x = (J'*J + lambda * Diagonal(DtD) ) \ (-J'*f_x)
    At_mul_B!(state.n_matrix, J, J)
    @simd for i in 1:n
        @inbounds state.n_matrix[i, i] += state.lambda * DtD[i]
    end
    At_mul_B!(state.n_vector, J, f_x)
    scale!(state.n_vector, -1)
    delta_x = state.n_matrix \ state.n_vector

    # apply box constraints
    if !isempty(lower)
        @simd for i in 1:n
           @inbounds delta_x[i] = max(x[i] + delta_x[i], lower[i]) - x[i]
        end
    end

    if !isempty(upper)
        @simd for i in 1:n
           @inbounds delta_x[i] = min(x[i] + delta_x[i], upper[i]) - x[i]
        end
    end

    # if the linear assumption is valid, our new residual should be:
    predicted_residual = sum(abs2, J * delta_x + f_x)

    # try the step and compute its quality
    trial_f = value(d, x + delta_x)
    trial_residual = sum(abs2, trial_f)

    # step quality = residual change / predicted residual change
    rho = (trial_residual - residual) / (predicted_residual - residual)

    if rho > method.min_step_quality
        state.previous_x = state.x
        state.previous_f_x = f_x
        state.x += delta_x
        if rho > method.good_step_quality
            # increase trust region radius
            state.lambda = max(method.lambda_decrease * state.lambda, MIN_LAMBDA)
        end
        value_gradient!!(d, state.x)
    else
        # decrease trust region radius
        state.lambda = min(method.lambda_increase * state.lambda, MAX_LAMBDA)
    end
end

function assess_convergence(d, options, state::LevenbergMarquardtState)
    # check convergence criteria:
    # 1. Small gradient: norm(J^T * f_x, Inf) < g_tol
    # 2. Small step size: norm(delta_x) < x_tol * x
    f_x = value(d)
    J = gradient(d)
    delta_x = state.x - state.previous_x
    delta_f_x = f_x - state.f_x_previous

    if norm(delta_x) < options.x_tol * (options.x_tol + norm(state.x))
        x_converged = true
    end

    if abs(delta_f_x) <=  options.f_tol * abs(f_x)
        f_converged = true
    end

    if delta_f_x > 0
        f_increased = true
    end

    if norm(J' * f_x, Inf) < options.g_tol
        g_converged = true
    end

    converged = x_converged || g_converged

    x_converged, f_converged, f_increased, g_converged, converged
end

function trace!(tr, d, state, iteration, method::LevenbergMarquardt, options)
    dt = Dict()
    if options.extended_trace
        dt["p"] = copy(state.x)
        dt["J(p)"] = copy(value(d))
    end
    J_norm = vecnorm(value(d), Inf)
    update!(tr,
            iteration,
            value(d),
            J_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end
