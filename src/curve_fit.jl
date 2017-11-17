immutable LsqFitResult{T,N}
    # simple type container for now, but can be expanded later
    model::Function
    dof::Int
    param::Vector{T}
    resid::Vector{T}
    rsquared::Float64
    stderror::Float64
    jacobian::Matrix{T}
    converged::Bool
    wt::Array{T,N}
end

# provide a method for those who have their own Jacobian function
function lmfit(model::Function, xpts::AbstractArray, ydata::AbstractArray, f::Function, g::Function, p0, wt; kwargs...)
    results = levenberg_marquardt(f, g, p0; kwargs...)
    p = minimizer(results)
    resid = f(p)
    prediction = model(xpts, p)
    rsquared = sum((prediction - mean(ydata)).^2) / sum((ydata - mean(ydata)).^2)
    stderror = sqrt(sum((prediction - resid).^2) / length(prediction))
    dof = length(resid) - length(p)
    return LsqFitResult(model, dof, p, resid, rsquared, stderror, g(p), converged(results), wt)
end

function lmfit(model::Function, xpts::AbstractArray, ydata::AbstractArray, f::Function, p0, wt; kwargs...)
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
    lmfit(model, xpts, ydata, f, g, p0, wt; kwargs...)
end

function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, p0; kwargs...)
    # construct the cost function
    f(p) = model(xpts, p) - ydata
    T = eltype(ydata)
    lmfit(model, xpts, ydata, f, p0, T[]; kwargs...)
end

function curve_fit(model::Function, jacobian_model::Function,
            xpts::AbstractArray, ydata::AbstractArray, p0; kwargs...)
    f(p) = model(xpts, p) - ydata
    g(p) = jacobian_model(xpts, p)
    T = eltype(ydata)
    lmfit(model, xpts, ydata, f, g, p0, T[]; kwargs...)
end

function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, wt::Vector, p0; kwargs...)
    # construct a weighted cost function, with a vector weight for each ydata
    # for example, this might be wt = 1/sigma where sigma is some error term
    f(p) = wt .* ( model(xpts, p) - ydata )
    lmfit(model, xpts, ydata, f, p0, wt; kwargs...)
end

function curve_fit(model::Function, jacobian_model::Function,
            xpts::AbstractArray, ydata::AbstractArray, wt::Vector, p0; kwargs...)
    f(p) = wt .* ( model(xpts, p) - ydata )
    g(p) = wt .* ( jacobian_model(xpts, p) )
    lmfit(model, xpts, ydata, f, g, p0, wt; kwargs...)
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
    lmfit(model, xpts, ydata, f, p0, wt; kwargs...)
end

function curve_fit(model::Function, jacobian_model::Function,
            xpts::AbstractArray, ydata::AbstractArray, wt::Matrix, p0; kwargs...)
    u = chol(wt)

    f(p) = u * ( model(xpts, p) - ydata )
    g(p) = u * ( jacobian_model(xpts, p) )
    lmfit(model, xpts, ydata, f, g, p0, wt; kwargs...)
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

function estimate_errors(fit::LsqFitResult, alpha=0.95; rtol::Real=NaN, atol::Real=0)
    # computes (1-alpha) error estimates from
    #   fit   : a LsqFitResult from a curve_fit()
    #   alpha : alpha percent confidence interval, (e.g. alpha=0.95 for 95% CI)
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    covar = estimate_covar(fit)

    # then the standard errors are given by the sqrt of the diagonal
    vars = diag(covar)
    vratio = minimum(vars)/maximum(vars)
    if !isapprox(vratio, 0.0, atol=atol, rtol=isnan(rtol) ? Base.rtoldefault(vratio, 0.0, 0) : rtol) && vratio < 0.0
        error("Covariance matrix is negative for atol=$atol and rtol=$rtol")
    end
    std_error = sqrt.(abs.(vars))

    # scale by quantile of the student-t distribution
    dist = TDist(fit.dof)
    return std_error * quantile(dist, alpha)
end
