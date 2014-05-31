CurveFit.jl
===========

The CurveFit package is a small library that provides basic curve fitting methods in pure Julia under an MIT license. It was ripped from the [Optim.jl](https://github.com/JuliaOpt/Optim.jl) library, which is where it's life began. 

# Basic Usage 

There are top-level methods `curve_fit()` and `estimate_errors()` that are useful for fitting data to non-linear models. See the following example:

    using CurveFit

    # a two-parameter exponential model
    model(xpts, p) = p[1]*exp(-xpts.*p[2])
    
    # some example data
    xpts = linspace(0,10,20)
    data = model(xpts, [1.0 2.0]) + 0.01*randn(length(xpts))
    
    beta, r, J = curve_fit(model, xpts, data, [0.5, 0.5])
	# beta = best fit parameters
	# r = vector of residuals
	# J = estimated Jacobian at solution
    
    # We can use these values to estimate errors on the fit parameters. To get 95% confidence error bars:
    errors = estimate_errors(beta, r, J)
    
# Existing Functions

* Curve Fitting: `curve_fit()` and `estimate_errors()`
* Levenberg-Marquardt: `levenberg_marquardt()`

