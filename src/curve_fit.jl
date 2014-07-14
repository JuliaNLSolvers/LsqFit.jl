
immutable LsqFitResult{T}
	# simple type container for now, but can be expanded later
	dof::Int
	param::Vector{T}
	resid::Vector{T}
	jacobian::Matrix{T}
end

function lmfit(f::Function, p0; kwargs...)
	# this is a convenience function for the curve_fit() methods
	# which assume f(p) is the cost functionj i.e. the residual of a
	# model where
	#   model(xpts, params...) = ydata + error (noise)

	# this minimizes f(p) using a least squares  sum of squared error:
	#   sse = sum(f(p)^2)
	# This is currently embedded in Optim.levelberg_marquardt()
	# which calls Optim.sse()
	#
	# returns p, f(p), g(p) where
	#   p    : best fit parameters
	#   f(p) : function evaluated at best fit p, (weighted) residuals
	#   g(p) : estimated Jacobian at p (Jacobian with respect to p)

	# construct Jacobian function, which uses finite difference method
	g = Calculus.jacobian(f)

	results = Optim.levenberg_marquardt(f, g, p0; kwargs...)
	p = results.minimum
	resid = f(p)
	dof = length(resid) - length(p)
	return LsqFitResult(dof, p, f(p), g(p))
end

function curve_fit(model::Function, xpts, ydata, p0; kwargs...)
	# construct the cost function
	f(p) = model(xpts, p) - ydata
	lmfit(f,p0; kwargs...)
end

function curve_fit(model::Function, xpts, ydata, wt::Vector, p0; kwargs...)
	# construct a weighted cost function, with a vector weight for each ydata
	# for example, this might be wt = 1/sigma where sigma is some error term
	f(p) = wt .* ( model(xpts, p) - ydata )
	lmfit(f,p0; kwargs...)
end

function curve_fit(model::Function, xpts, ydata, wt::Matrix, p0; kwargs...)
	# as before, construct a weighted cost function with where this
	# method uses a matrix weight.
	# for example: an inverse_covariance matrix

	# Cholesky is effectively a sqrt of a matrix, which is what we want
	# to minimize in the least-squares of levenberg_marquardt()
	# This requires the matrix to be positive definite
	u = chol(wt)

	f(p) = u * ( model(xpts, p) - ydata )
	lmfit(f,p0; kwargs...)
end

function estimate_covar(fit::LsqFitResult)
	# computes covariance matrix of fit parameters
	r = fit.resid
	J = fit.jacobian

	# mean square error is: standard sum square error / degrees of freedom
	mse = Optim.sse(r) / fit.dof

	# compute the covariance matrix from the QR decomposition
	Q,R = qr(J)
	Rinv = inv(R)
	covar = Rinv*Rinv'*mse
end

function estimate_errors(fit::LsqFitResult, alpha=0.95)
	# computes (1-alpha) error estimates from
	#   fit   : a LsqFitResult from a curve_fit()
	#   alpha : alpha percent confidence interval, (e.g. alpha=0.95 for 95% CI)
	covar = estimate_covar(fit)

	# then the standard errors are given by the sqrt of the diagonal
	std_error = sqrt(diag(covar))

	# scale by quantile of the student-t distribution
	dist = TDist(fit.dof)
	std_error *= quantile(dist, alpha)
end
