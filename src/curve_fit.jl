immutable LsqFitResult{T,N}
    # simple type container for now, but can be expanded later
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
    curve_fit(model, [jacobian], x, y, [w,] p0; kwargs...)

Fit data to a non-linear `model` by minimizing the (weighted) residual between the model and the dependent variable data (`y`). `p0` is an initial model parameter guess.

The weight (`w`) can be neglected to perform an unweighted fit. An unweighted fit is the numerical equivalent of `w=1` for each point, though unweighted error estimates are handled differently from weighted error estimates even when the weights are uniform.

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
* `w`: (optional) weight applied to the residual; can be a vector (of `length(x)` size or empty) or matrix (inverse covariance matrix)
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

function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, wt::Vector, p0; kwargs...)
    # construct a weighted cost function, with a vector weight for each ydata
    # for example, this might be wt = 1/sigma where sigma is some error term
    f(p) = wt .* ( model(xpts, p) - ydata )
    lmfit(f,p0,wt; kwargs...)
end

function curve_fit(model::Function, jacobian_model::Function,
            xpts::AbstractArray, ydata::AbstractArray, wt::Vector, p0; kwargs...)
    f(p) = wt .* ( model(xpts, p) - ydata )
    g(p) = wt .* ( jacobian_model(xpts, p) )
    lmfit(f, g, p0, wt; kwargs...)
end

function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, wt::Matrix, p0; kwargs...)
    # as before, construct a weighted cost function with where this
    # method uses a matrix weight.
    # for example: an inverse_covariance matrix

    # Cholesky is effectively a sqrt of a matrix, which is what we want
    # to minimize in the least-squares of levenberg_marquardt()
    # This requires the matrix to be positive definite
    u = chol(wt)

    f(p) = u * ( model(xpts, p) - ydata )
    lmfit(f,p0,wt; kwargs...)
end

function curve_fit(model::Function, jacobian_model::Function,
            xpts::AbstractArray, ydata::AbstractArray, wt::Matrix, p0; kwargs...)
    u = chol(wt)

    f(p) = u * ( model(xpts, p) - ydata )
    g(p) = u * ( jacobian_model(xpts, p) )
    lmfit(f, g, p0, wt; kwargs...)
end

"""
    estimate_covar(fit)

Computes the covariance matrix of fit parameters from the jacobian of the model at the fit point, using the weights (if specified) as the inverse covariance of observations.
"""
function estimate_covar(fit::LsqFitResult)
    J = fit.jacobian
    r = fit.resid
    # σ^2 = sum of squared error / degrees of freedom
    sigma_sqr = sum(abs2, r) / fit.dof

    if isempty(fit.wt)
        # compute the covariance matrix from the QR decomposition
        Q,R = qr(J)
        Rinv = inv(R)
        covar = Rinv*Rinv'*sigma_sqr
    elseif length(size(fit.wt)) == 1
        # compute the covariance matrix from the Jacobian and a vector weight
        covar = inv(J'*Diagonal(fit.wt)*J)*sigma_sqr
    else
        # compute the covariance matrix from the Jacobian and a matrix weights
        covar = inv(J'*fit.wt*J)*sigma_sqr

    end

    return covar
end

"""
    standard_error(fit; rtol=NaN, atol=0)

Compute the standard error of parameter estimates from `LsqFitResult`.

If no weights are provided for the fits, the variance is estimated from the squared error of the fits / degress of freedom. If weights are provided, the weights are assumed to be the inverse of the variances or of the covariance matrix, and errors are estimated based on these and the jacobian, assuming a linearization of the model around the minimum squared error point.

!!! note

    The result is already scaled by the associated degrees of freedom. It is also a LOCAL quantity calculated from the jacobian of the model evaluated at the best fit point and NOT the result of a parameter exploration.

# Arguments
* `fit::LsqFitResult`: a LsqFitResult from curve_fit()
* `rtol::Real=NaN`: relative tolerance for approximate comparisson to 0.0 in negativity check
* `atol::Real=0`: absolute tolerance for approximate comparisson to 0.0 in negativity check

# Example
```julia
julia> fit = curve_fit(model, xdata, ydata, p0)
julia> sigma = standard_error(fit)
2-element Array{Float64,1}:
 0.0114802
 0.0520416
```
"""
function standard_error(fit::LsqFitResult; rtol::Real=NaN, atol::Real=0)
    covar = estimate_covar(fit)
    # then the standard errors are given by the sqrt of the diagonal
    vars = diag(covar)
    vratio = minimum(vars)/maximum(vars)
    if !isapprox(vratio, 0.0, atol=atol, rtol=isnan(rtol) ? Base.rtoldefault(vratio, 0.0, 0) : rtol) && vratio < 0.0
        error("Covariance matrix is negative for atol=$atol and rtol=$rtol")
    end
    return sqrt.(abs.(vars))
end

"""
    margin_error(fit, alpha=0.05; rtol=NaN, atol=0)

Return the product of standard error and critical value of each parameter at `alpha` significance level.

# Arguments
* `fit::LsqFitResult`: a LsqFitResult from `curve_fit()`
* `alpha=0.05` : significance level, e.g. alpha=0.05 for 95% confidence
* `rtol::Real=NaN`: relative tolerance for approximate comparisson to 0.0 in negativity check
* `atol::Real=0`: absolute tolerance for approximate comparisson to 0.0 in negativity check

# Example
```julia
julia> fit = curve_fit(model, xdata, ydata, p0)
julia> margin_of_error = margin_error(fit, 0.1)
2-element Array{Float64,1}:
 0.0199073
 0.0902435
```
"""
function margin_error(fit::LsqFitResult, alpha=0.05; rtol::Real=NaN, atol::Real=0)
    std_errors = standard_error(fit; rtol=rtol, atol=atol)
    dist = TDist(fit.dof)
    critical_values = quantile(dist, 1 - alpha/2)
    # scale standard errors by quantile of the student-t distribution (critical values)
    return std_errors * critical_values
end

"""
    confidence_interval(fit, alpha=0.05; rtol=NaN, atol=0)

Return confidence interval of each parameter at `alpha` significance level.

# Arguments
* `fit::LsqFitResult`: a LsqFitResult from `curve_fit()`
* `alpha=0.05` : significance level, e.g. alpha=0.05 for 95% confidence
* `rtol::Real=NaN`: relative tolerance for approximate comparisson to 0.0 in negativity check
* `atol::Real=0`: absolute tolerance for approximate comparisson to 0.0 in negativity check

# Example
```julia
julia> fit = curve_fit(model, xdata, ydata, p0)
julia> confidence_interval = confidence_interval(fit, 0.1)
2-element Array{Tuple{Float64,Float64},1}:
 (0.976316, 1.01613)
 (1.91047, 2.09096)
```
"""
function confidence_interval(fit::LsqFitResult, alpha=0.05; rtol::Real=NaN, atol::Real=0)
    std_errors = standard_error(fit; rtol=rtol, atol=atol)
    margin_of_errors = margin_error(fit, alpha; rtol=rtol, atol=atol)
    confidence_intervals = collect(zip(fit.param-margin_of_errors, fit.param+margin_of_errors))
end

@deprecate estimate_errors(fit::LsqFitResult, confidence=0.95; rtol::Real=NaN, atol::Real=0) margin_error(fit, 1-confidence; rtol=rtol, atol=atol)
