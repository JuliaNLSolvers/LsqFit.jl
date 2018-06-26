immutable LsqFitResult{T,N}
    dof::Int
    param::Vector{T}
    resid::Vector{T}
    jacobian::Matrix{T}
    converged::Bool
    wt::Array{T,N}
end

# provide a method for those who have their own Jacobian function
function lmfit(f::Function, g::Function, p0, wt; kwargs...)
    results = levenberg_marquardt(f, g, p0; kwargs...)
    p = minimizer(results)
    resid = f(p)
    dof = length(resid) - length(p)
    return LsqFitResult(dof, p, f(p), g(p), converged(results), wt)
end

function lmfit(f::Function, p0, wt; kwargs...)
    # this is a convenience function for the curve_fit() methods
    # which assume f(p) is the cost functionj i.e. the residual of a
    # model where
    #   model(xpts, params...) = ydata + error (noise)

    # this minimizes f(p) using a least squares sum of squared error:
    #   sse = sum(f(p)^2)
    # This is currently embedded in Optim.levelberg_marquardt()
    # which calls sum(abs2)
    #
    # returns p, f(p), g(p) where
    #   p    : best fit parameters
    #   f(p) : function evaluated at best fit p, (weighted) residuals
    #   g(p) : estimated Jacobian at p (Jacobian with respect to p)

    # construct Jacobian function, which uses finite difference method
    g = Calculus.jacobian(f)
    lmfit(f, g, p0, wt; kwargs...)
end


"""
    curve_fit(model, [jacobian], x, y, [sigma,] p0; kwargs...)

Fit data to a non-linear `model` by minimizing the (transformed) residual between the model and the dependent variable data (`y`). `p0` is an initial model parameter guess.

The return object is a composite type (`LsqFitResult`), with some interesting values:

* `fit.dof` : degrees of freedom
* `fit.param` : best fit parameters
* `fit.resid` : vector of residuals
* `fit.jacobian` : estimated Jacobian at solution

# Arguments
* `model`: function that takes two arguments (x, params)
* `jacobian`: (optional) function that returns the Jacobian matrix of `model`
* `x`: the independent variable
* `y`: the dependent variable that constrains `model`
* `sigma::Vector`: (optional) the standard deviations of errors to perform Weighted Least Squares.
* `sigma::Matrix`: (optional) the covariance matrix of errors to perform General Least Squares.
* `p0`: initial guess of the model parameters
* `kwargs`: tuning parameters for fitting, passed to `levenberg_marquardt`, such as `maxIter` or `show_trace`

# Example
```julia
# a two-parameter exponential model
# t: array of independent variables
# p: array of model parameters
julia> model(x, p) = p[1] * exp.(-p[2] * t)

# some example data
# tdata: independent variables
# ydata: dependent variable
julia> tdata = linspace(0,10,20)
julia> ydata = model(tdata, [1.0 2.0]) + 0.01*randn(length(tdata))
julia> p0 = [0.5, 0.5]

julia> fit = curve_fit(model, tdata, ydata, p0)
```
"""
function curve_fit end

function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, p0; kwargs...)
    # construct the cost function
    f(p) = model(xpts, p) - ydata
    T = eltype(ydata)
    lmfit(f,p0,T[]; kwargs...)
end

function curve_fit(model::Function, jacobian_model::Function,
                   xpts::AbstractArray, ydata::AbstractArray, p0; kwargs...)
    f(p) = model(xpts, p) - ydata
    g(p) = jacobian_model(xpts, p)
    T = eltype(ydata)
    lmfit(f, g, p0, T[]; kwargs...)
end

"""
    curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, sigma::Vector, p0; kwargs...)

use `sigma` to construct a weighted cost function to perform Weighted Least Squares, where `sigma` is a vector of the standard deviations of errors, i.e. ϵ_i ~ N(0, σ_i^2), which could be estimated as `abs(fit.resid)`.
"""
function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, sigma::Vector, p0; kwargs...)
    warn("The `weight` argument has been deprecated. Please make sure that you're passing the vector of the standard deviations of errors, which could be estimated by `abs(fit.resid)`.")
    sqrt_wt = 1 ./ sigma
    wt = sqrt_wt.^2
    f(p) = sqrt_wt .* ( model(xpts, p) - ydata )
    lmfit(f,p0,wt; kwargs...)
end

function curve_fit(model::Function, jacobian_model::Function,
                   xpts::AbstractArray, ydata::AbstractArray, sigma::Vector, p0; kwargs...)
    warn("The `weight` argument has been deprecated. Please make sure that you're passing the vector of the standard deviations of errors, which could be estimated by `abs(fit.resid)`.")
    sqrt_wt = 1 ./ sigma
    wt = sqrt_wt.^2
    f(p) = sqrt_wt .* ( model(xpts, p) - ydata )
    g(p) = sqrt_wt .* ( jacobian_model(xpts, p) )
    lmfit(f, g, p0, wt; kwargs...)
end

"""
    function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, sigma::Matrix, p0; kwargs...)

use `sigma` to construct a transformed cost function to perform General Least Squares, where `sigma` is a matrix of the covariance matrix of error, i.e. ϵ ~ N(0, Σ).
"""
function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, sigma::Matrix, p0; kwargs...)
    warn("The `weight` argument has been deprecated. Please make sure that you're passing the covariance matrix of errors.")
    wt = inv(sigma)
    # Cholesky is effectively a sqrt of a matrix, which is what we want
    # to minimize in the least-squares of levenberg_marquardt()
    # This requires the matrix to be positive definite
    u = chol(wt)
    f(p) = u * ( model(xpts, p) - ydata )
    lmfit(f,p0,wt; kwargs...)
end

function curve_fit(model::Function, jacobian_model::Function,
                   xpts::AbstractArray, ydata::AbstractArray, sigma::Matrix, p0; kwargs...)
    warn("The `weight` argument has been deprecated. Please make sure that you're passing the covariance matrix of errors.")
    wt = inv(sigma)
    u = chol(wt)
    f(p) = u * ( model(xpts, p) - ydata )
    g(p) = u * ( jacobian_model(xpts, p) )
    lmfit(f, g, p0, wt; kwargs...)
end

function estimate_covar(fit::LsqFitResult)
    # computes covariance matrix of fit parameters
    J = fit.jacobian

    if isempty(fit.wt)
        r = fit.resid

        # mean square error is: standard sum square error / degrees of freedom
        mse = sum(abs2, r) / fit.dof

        # compute the covariance matrix from the QR decomposition
        Q,R = qr(J)
        Rinv = inv(R)
        covar = Rinv*Rinv'*mse
    elseif length(size(fit.wt)) == 1
        covar = inv(J'*Diagonal(fit.wt)*J)
    else
        covar = inv(J'*fit.wt*J)
    end

    return covar
end

function standard_error(fit::LsqFitResult; rtol::Real=NaN, atol::Real=0)
    # computes standard error of estimates from
    #   fit   : a LsqFitResult from a curve_fit()
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    covar = estimate_covar(fit)
    # then the standard errors are given by the sqrt of the diagonal
    vars = diag(covar)
    vratio = minimum(vars)/maximum(vars)
    if !isapprox(vratio, 0.0, atol=atol, rtol=isnan(rtol) ? Base.rtoldefault(vratio, 0.0, 0) : rtol) && vratio < 0.0
        error("Covariance matrix is negative for atol=$atol and rtol=$rtol")
    end
    return sqrt.(abs.(vars))
end

function margin_error(fit::LsqFitResult, alpha=0.05; rtol::Real=NaN, atol::Real=0)
    # computes margin of error at alpha significance level from
    #   fit   : a LsqFitResult from a curve_fit()
    #   alpha : significance level, e.g. alpha=0.05 for 95% confidence
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    std_errors = standard_error(fit; rtol=rtol, atol=atol)
    dist = TDist(fit.dof)
    critical_values = quantile(dist, 1 - alpha/2)
    # scale standard errors by quantile of the student-t distribution (critical values)
    return std_errors * critical_values
end

function confidence_interval(fit::LsqFitResult, alpha=0.05; rtol::Real=NaN, atol::Real=0)
    # computes confidence intervals at alpha significance level from
    #   fit   : a LsqFitResult from a curve_fit()
    #   alpha : significance level, e.g. alpha=0.05 for 95% confidence
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    std_errors = standard_error(fit; rtol=rtol, atol=atol)
    margin_of_errors = margin_error(fit, alpha; rtol=rtol, atol=atol)
    confidence_intervals = collect(zip(fit.param-margin_of_errors, fit.param+margin_of_errors))
end

@deprecate estimate_errors(fit::LsqFitResult, confidence=0.95; rtol::Real=NaN, atol::Real=0) margin_error(fit, 1-confidence; rtol=rtol, atol=atol)
