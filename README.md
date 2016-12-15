LsqFit.jl
===========

The LsqFit package is a small library that provides basic least-squares fitting in pure Julia under an MIT license. The basic functionality was originaly in [Optim.jl](https://github.com/JuliaOpt/Optim.jl), before being separated into this library.  At this time, `LsqFit` only utilizes the Levenberg-Marquardt algorithm for non-linear fitting.

[![Build Status](https://travis-ci.org/JuliaOpt/LsqFit.jl.svg)](https://travis-ci.org/JuliaOpt/LsqFit.jl)

[![LsqFit](http://pkg.julialang.org/badges/LsqFit_0.3.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.3)
[![LsqFit](http://pkg.julialang.org/badges/LsqFit_0.4.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.4)
[![LsqFit](http://pkg.julialang.org/badges/LsqFit_0.5.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.5)

Basic Usage
-----------

There are top-level methods `curve_fit()` and `estimate_errors()` that are useful for fitting data to non-linear models. See the following example:

    using LsqFit

    # a two-parameter exponential model
    # x: array of independent variables
    # p: array of model parameters
    model(x, p) = p[1]*exp(-x.*p[2])

    # some example data
    # xdata: independent variables
    # ydata: dependent variable
    xdata = linspace(0,10,20)
    ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))

    fit = curve_fit(model, xdata, ydata, [0.5, 0.5])
    # fit is a composite type (LsqFitResult), with some interesting values:
    #	fit.dof: degrees of freedom
    #	fit.param: best fit parameters
    #	fit.resid: residuals = vector of residuals
    #	fit.jacobian: estimated Jacobian at solution

    # We can estimate errors on the fit parameters,
    # to get 95% confidence error bars:
    errors = estimate_errors(fit, 0.95)


Existing Functionality
----------------------

`fit = curve_fit(model, x, y, w, p0; kwargs...)`:

* `model`: function that takes two arguments (x, params)
* `x`: the independent variable
* `y`: the dependent variable that constrains `model`
* `w`: weight applied to the residual; can be a vector (of `length(x)` size) or matrix (inverse covariance)
* `p0`: initial guess of the model parameters
* `kwargs`: tuning parameters for fitting, passed to [`levenberg_marquardt`](https://github.com/JuliaOpt/Optim.jl/blob/master/src/levenberg_marquardt.jl) of `Optim.jl`, such as `maxIter` or `show_trace`
* `fit`: composite type of results (`LsqFitResult`)


This performs a fit using a non-linear iteration to minimize the (weighted) residual between the model and the dependent variable data (`y`). The weight (`w`) can be neglected (as per the example) to perform an unweighted fit. An unweighted fit is the numerical equivalent of `w=1` for each point.

----

`sigma = estimate_errors(fit, alpha=0.95)`:

* `fit`: result of curve_fit (a `LsqFitResult` type)
* `alpha`: confidence limit to calculate for the errors on parameters
* `sigma`: typical (symmetric) standard deviation for each parameter

This returns the error or uncertainty of each parameter fit to the model and already scaled by the associated degrees of freedom.  Please note, this is a LOCAL quantity calculated from the jacobian of the model evaluated at the best fit point and NOT the result of a parameter exploration.

----

`covar = estimate_covar(fit)`:

* `fit`: result of curve_fit (a `LsqFitResult` type)
* `covar`: parameter covariance matrix calculated from the jacobian of the model at the fit point

This returns the parameter covariance matrix evaluted at the best fit point.
