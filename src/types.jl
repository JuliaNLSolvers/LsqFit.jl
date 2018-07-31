abstract type AbstractOptimizer end
abstract type AbstractOptimizerState end
abstract type OptimizationResults end

struct Options{T, TCallback}
    x_tol::T
    f_tol::T
    g_tol::T
    f_calls_limit::Int
    g_calls_limit::Int
    h_calls_limit::Int
    allow_f_increases::Bool
    iterations::Int
    store_trace::Bool
    show_trace::Bool
    extended_trace::Bool
    show_every::Int
    callback::TCallback
    time_limit::Float64
end

function Options(;x_tol::Real = 1e-8,
                  f_tol::Real = 0.0,
                  g_tol::Real = 1e-12,
                  f_calls_limit::Int = 0,
                  g_calls_limit::Int = 0,
                  h_calls_limit::Int = 0,
                  allow_f_increases::Bool = false,
                  iterations::Int = 100,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  show_every::Int = 1,
                  callback = nothing,
                  time_limit::AbstractFloat = NaN)
    show_every = show_every > 0 ? show_every : 1
    Options(x_tol,
            f_tol,
            g_tol,
            f_calls_limit,
            g_calls_limit,
            h_calls_limit,
            allow_f_increases,
            iterations,
            store_trace,
            show_trace,
            extended_trace,
            show_every,
            callback,
            time_limit)
end

function print_header(method::AbstractOptimizer)
        @printf "Iter     Function value   Jacobian norm \n"
end

struct OptimizationState{Tf, T <: AbstractOptimizer}
    iteration::Int
    value::Tf
    g_norm::Tf
    metadata::Dict
end

const OptimizationTrace{Tf, T} = Vector{OptimizationState{Tf, T}}

immutable MultivariateOptimizationResults{O, T, Tx, Tc, Tf, M} <: OptimizationResults
    method::O
    initial_x::Tx
    minimizer::Tx
    minimum::Tf
    iterations::Int
    iteration_converged::Bool
    x_converged::Bool
    x_tol::T
    x_abschange::Tc
    f_converged::Bool
    f_tol::T
    f_abschange::Tc
    g_converged::Bool
    g_tol::T
    g_residual::Tc
    f_increased::Bool
    trace::M
    f_calls::Int
    g_calls::Int
    h_calls::Int
end

immutable LsqFitResult{T,N}
    n::Int
    dof::Int
    param::Vector{T}
    ydata::Vector{T}
    resid::Vector{T}
    jacobian::Matrix{T}
    wt::Array{T,N}
    algorithm::String
    iterations::Int
    converged::Bool
end

function Base.append!(a::MultivariateOptimizationResults, b::MultivariateOptimizationResults)
    a.iterations += iterations(b)
    a.minimizer = minimizer(b)
    a.minimum = minimum(b)
    a.iteration_converged = iteration_limit_reached(b)
    a.x_converged = x_converged(b)
    a.f_converged = f_converged(b)
    a.g_converged = g_converged(b)
    append!(a.trace, b.trace)
    a.f_calls += f_calls(b)
    a.g_calls += g_calls(b)
end

function Base.show(io::IO, t::OptimizationState)
    @printf io "%6d   %14e   %14e\n" t.iteration t.value t.g_norm
    if !isempty(t.metadata)
        for (key, value) in t.metadata
            @printf io " * %s: %s\n" key value
        end
    end
    return
end

function Base.show(io::IO, tr::OptimizationTrace)
    @printf io "Iter     Function value   Jacobian norm \n"
    @printf io "------   --------------   ------------- \n"
    for state in tr
        show(io, state)
    end
    return
end

function Base.show(io::IO, r::MultivariateOptimizationResults)
    first_two(fr) = [x for (i, x) in enumerate(fr)][1:2]

    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" summary(r)
    if length(join(initial_state(r), ",")) < 40
        @printf io " * Starting Point: [%s]\n" join(initial_state(r), ",")
    else
        @printf io " * Starting Point: [%s, ...]\n" join(first_two(initial_state(r)), ",")
    end
    if length(join(minimizer(r), ",")) < 40
        @printf io " * Minimizer: [%s]\n" join(minimizer(r), ",")
    else
        @printf io " * Minimizer: [%s, ...]\n" join(first_two(minimizer(r)), ",")
    end
    @printf io " * Minimum: %e\n" minimum(r)
    @printf io " * Iterations: %d\n" iterations(r)
    @printf io " * Convergence: %s\n" converged(r)
    if isa(r.method, NelderMead)
        @printf io "   *  √(Σ(yᵢ-ȳ)²)/n < %.1e: %s\n" g_tol(r) g_converged(r)
    else
        @printf io "   * |x - x'| ≤ %.1e: %s \n" x_tol(r) x_converged(r)
        @printf io "     |x - x'| = %.2e \n"  x_abschange(r)
        @printf io "   * |f(x) - f(x')| ≤ %.1e |f(x)|: %s\n" f_tol(r) f_converged(r)
        @printf io "     |f(x) - f(x')| = %.2e |f(x)|\n" f_relchange(r)
        @printf io "   * |g(x)| ≤ %.1e: %s \n" g_tol(r) g_converged(r)
        @printf io "     |g(x)| = %.2e \n"  g_residual(r)
        @printf io "   * Stopped by an increasing objective: %s\n" (f_increased(r) && !iteration_limit_reached(r))
    end
    @printf io "   * Reached Maximum Number of Iterations: %s\n" iteration_limit_reached(r)
    @printf io " * Objective Calls: %d" f_calls(r)
    if !(isa(r.method, NelderMead) || isa(r.method, SimulatedAnnealing))
        @printf io "\n * Gradient Calls: %d" g_calls(r)
    end
    if isa(r.method, Newton) || isa(r.method, NewtonTrustRegion)
        @printf io "\n * Hessian Calls: %d" h_calls(r)
    end
    return
end

function Base.show(io::IO, fit::LsqFitResult)
    print(io,
    """Results of Least Squares Fitting:
    * Algorithm: $(fit.algorithm)
    * Iterations: $(fit.iterations)
    * Converged: $(fit.converged)
    * Estimated Parameters: $(fit.param)
    * Sample Size: $(fit.n)
    * Degrees of Freedom: $(fit.dof)
    * Weights: $(fit.wt)
    * Sum of Squared Errors: $(round(sse(fit), 4))
    * Mean Squared Errors: $(round(mse(fit), 4))
    * R²: $(round(r2(fit), 4))
    * Adjusted R²: $(round(adjr2(fit), 4))
    """)
    println(io, "\nVariance Inferences:")
    nc = 4
    nr = length(fit.param)
    outrows = Matrix{String}(nr+1, nc)
    outrows[1, :] = ["k", "value", "std error", "95% conf int"]

    for i in 1:nr
        outrows[i+1, :] = ["$i", "$(round(fit.param[i], 4))", "$(round(standard_error(fit)[i], 4))",
                           "$(round.(confidence_interval(fit)[i], 3))"]
    end

    colwidths = length.(outrows)
    max_colwidths = [maximum(view(colwidths, :, i)) for i in 1:nc]

    for r in 1:nr+1
        for c in 1:nc
            cur_cell = outrows[r, c]
            cur_cell_len = length(cur_cell)

            padding = " "^(max_colwidths[c]-cur_cell_len)
            if c > 1
                padding = " "*padding
            end

            print(io, padding)
            print(io, cur_cell)
        end
        print(io, "\n")
    end
end
