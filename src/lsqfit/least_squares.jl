function least_squares(d::AbstractObjective, initial_x::AbstractArray,
                       method::AbstractOptimizer = LevenbergMarquardt(),
                       options::Options = Options(),
                       state = initial_state(d, initial_x, method, options))

    t0 = time() # Initial time stamp used to control early stopping by options.time_limit

    tr = OptimizationTrace{typeof(value(d)), typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback != nothing
    stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
    f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
    x_converged, f_converged, f_increased = false, false, false
    g_converged = vecnorm(gradient!(d, initial_x), Inf) < options.g_tol
    converged = g_converged

    # prepare iteration counter (used to make "initial state" trace entry)
    iteration = 0

    while !converged && !stopped && iteration < options.iterations
       iteration += 1
       update_state!(state, d, method)
       x_converged, f_converged, f_increased, g_converged, converged = assess_convergence(d, options, state)

       if tracing
           #TODO
           trace!(method, options)
           
       # Check time_limit; if none is provided it is NaN and the comparison
       # will always return false.
       stopped_by_time_limit = time()-t0 > options.time_limit
       f_limit_reached = options.f_calls_limit > 0 && f_calls(d) >= options.f_calls_limit ? true : false
       g_limit_reached = options.g_calls_limit > 0 && g_calls(d) >= options.g_calls_limit ? true : false
       h_limit_reached = options.h_calls_limit > 0 && h_calls(d) >= options.h_calls_limit ? true : false

       if (f_increased && !options.allow_f_increases) || stopped_by_callback ||
           stopped_by_time_limit || f_limit_reached || g_limit_reached || h_limit_reached
           stopped = true
       end
    end

    # we can just check minimum, as we've earlier enforced same types/eltypes
    # in variables besides the option settings
    T = typeof(options.f_tol)
    f_incr_pick = f_increased && !options.allow_f_increases

    return MultivariateOptimizationResults(method,
                                           initial_x,
                                           pick_best_x(f_incr_pick, state),
                                           pick_best_f(f_incr_pick, state, d),
                                           iteration,
                                           iteration == options.iterations,
                                           x_converged,
                                           T(options.x_tol),
                                           x_abschange(state),
                                           f_converged,
                                           T(options.f_tol),
                                           f_abschange(d, state),
                                           g_converged,
                                           T(options.g_tol),
                                           g_residual(d),
                                           f_increased,
                                           tr,
                                           f_calls(d),
                                           g_calls(d),
                                           h_calls(d))
end

function least_squares(f::Function, initial_x::AbstractArray, method::AbstractOptimizer = LevenbergMarquardt(), options::Options = Options(); inplace = true, autodiff = :finite, kwargs...)
    d = OnceDifferentiable(f, initial_x, inplace = inplace, autodiff = autodiff, kwargs...)
    least_squares(d, initial_x, method, options)
end

function least_squares(f::Function, j::Function, initial_x::AbstractArray, method::AbstractOptimizer = LevenbergMarquardt(), options::Options = Options(); inplace = true, autodiff = :finite, kwargs...)
    d = OnceDifferentiable(f, j, initial_x, inplace = inplace, kwargs...)
    least_squares(d, initial_x, method, options)
end
