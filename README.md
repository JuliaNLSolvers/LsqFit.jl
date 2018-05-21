LsqFit.jl
===========

The LsqFit package is a small library that provides basic least-squares fitting in pure Julia under an MIT license. The basic functionality was originaly in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), before being separated into this library.  At this time, `LsqFit` only utilizes the Levenberg-Marquardt algorithm for non-linear fitting.

[![Build Status](https://travis-ci.org/JuliaNLSolvers/LsqFit.jl.svg)](https://travis-ci.org/JuliaNLSolvers/LsqFit.jl)

[![LsqFit](http://pkg.julialang.org/badges/LsqFit_0.3.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.3)
[![LsqFit](http://pkg.julialang.org/badges/LsqFit_0.4.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.4)
[![LsqFit](http://pkg.julialang.org/badges/LsqFit_0.5.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.5)

Basic Usage
-----------

There are top-level methods `curve_fit()` and `estimate_errors()` that are useful for fitting data to non-linear models. See the following example:
```julia
using LsqFit

# a two-parameter exponential model
# x: array of independent variables
# p: array of model parameters
model(x, p) = p[1]*exp.(-x.*p[2])

# some example data
# xdata: independent variables
# ydata: dependent variable
xdata = linspace(0,10,20)
ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
p0 = [0.5, 0.5]

fit = curve_fit(model, xdata, ydata, p0)
# fit is a composite type (LsqFitResult), with some interesting values:
#	fit.dof: degrees of freedom
#	fit.param: best fit parameters
#	fit.resid: residuals = vector of residuals
#	fit.jacobian: estimated Jacobian at solution

# We can estimate errors on the fit parameters,
# to get standard error of each parameter:
sigma = standard_error(fit)
# to get margin of error and confidence interval of each parameter at 5% significance level:
margin_of_error = margin_error(fit, 0.05)
confidence_interval = confidence_interval(fit, 0.05)

# The finite difference method is used above to approximate the Jacobian.
# Alternatively, a function which calculates it exactly can be supplied instead.
function jacobian_model(x,p)
    J = Array{Float64}(length(x),length(p))
    J[:,1] = exp.(-x.*p[2])    #dmodel/dp[1]
    J[:,2] = -x.*p[1].*J[:,1]  #dmodel/dp[2]
    J
end
fit = curve_fit(model, jacobian_model, xdata, ydata, p0)
```

Existing Functionality
----------------------

`fit = curve_fit(model, [jacobian], x, y, [w,] p0; kwargs...)`:

* `model`: function that takes two arguments (x, params)
* `jacobian`: (optional) function that returns the Jacobian matrix of `model`
* `x`: the independent variable
* `y`: the dependent variable that constrains `model`
* `w`: (optional) weight applied to the residual; can be a vector (of `length(x)` size or empty) or matrix (inverse covariance matrix)
* `p0`: initial guess of the model parameters
* `kwargs`: tuning parameters for fitting, passed to `levenberg_marquardt`, such as `maxIter` or `show_trace`
* `fit`: composite type of results (`LsqFitResult`)


This performs a fit using a non-linear iteration to minimize the (weighted) residual between the model and the dependent variable data (`y`). The weight (`w`) can be neglected (as per the example) to perform an unweighted fit. An unweighted fit is the numerical equivalent of `w=1` for each point  (although unweighted error estimates are handled differently from weighted error estimates even when the weights are uniform).

----

`sigma = standard_error(fit; atol, rtol)`:

* `fit`: result of curve_fit (a `LsqFitResult` type)
* `atol`: absolute tolerance for negativity check
* `rtol`: relative tolerance for negativity check

This returns the error or uncertainty of each parameter fit to the model and already scaled by the associated degrees of freedom.  Please note, this is a LOCAL quantity calculated from the jacobian of the model evaluated at the best fit point and NOT the result of a parameter exploration.

If no weights are provided for the fits, the variance is estimated from the mean squared error of the fits. If weights are provided, the weights are assumed to be the inverse of the variances or of the covariance matrix, and errors are estimated based on these and the jacobian, assuming a linearization of the model around the minimum squared error point.

`margin_of_error = margin_error(fit, alpha=0.05; atol, rtol)`:

* `fit`: result of curve_fit (a `LsqFitResult` type)
* `alpha`: significance level
* `atol`: absolute tolerance for negativity check
* `rtol`: relative tolerance for negativity check

This returns the product of standard error and critical value of each parameter at `alpha` significance level.

`confidence_interval = confidence_interval(fit, alpha=0.05; atol, rtol)`:

* `fit`: result of curve_fit (a `LsqFitResult` type)
* `alpha`: significance level
* `atol`: absolute tolerance for negativity check
* `rtol`: relative tolerance for negativity check

This returns confidence interval of each parameter at `alpha` significance level.

----

`covar = estimate_covar(fit)`:

* `fit`: result of curve_fit (a `LsqFitResult` type)
* `covar`: parameter covariance matrix calculated from the jacobian of the model at the fit point, using the weights (if specified) as the inverse covariance of observations

This returns the parameter covariance matrix evaluted at the best fit point.
