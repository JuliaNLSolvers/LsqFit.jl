# Tutorial

## Estimate Non-linear Models

Assume the relationship between independent variables (``x``) and dependent variable (``Y``) follows

```math
Y_i = h(x_1^{(i)}, x_2^{(i)},  x_3^{(i)}...; \theta_1, \theta_2, \theta_3...)+\epsilon_i
```

or in matrix form

```math
Y = h(\mathrm{X}; \theta)+\epsilon
```

where ``h`` is a non-linear function depends on independent variables ``x`` and parameters ``\theta``. In order to find the parameters ``\theta`` that "best" fit our data, we choose the parameters ``\hat{\theta}`` that minimize the sum of squared residuals from our data.

```math
\hat{\theta} = \underset{\theta}{\mathrm{arg\,min}} \quad S(\hat{\theta})= [h(\mathrm{X}; \hat{\theta}) - Y_i]^2
```

Given that the function is non-linear, there's no analytical solutions and we have to seek computational tools to find the least squares solutions, which is the main focus of `LsqFit.jl`. To fit data using `LsqFit.jl`, pass the model defined, data and initial values to `curve_fit()`. For now, `LsqFit.jl` only supports levenberg marquardt algorithm.

```julia
julia> # t: array of independent variables
julia> # p: array of model parameters
julia> model(t, p) = p[1] * exp.(-p[2] * t)
julia> p0 = [0.5, 0.5]
julia> fit = curve_fit(model, tdata, ydata, p0)
```

It will return return a composite type (`LsqFitResult`), with some interesting values:

*	`fit.dof`: degrees of freedom
*	`fit.param`: best fit parameters
*	`fit.resid`: vector of residuals
*	`fit.jacobian`: estimated Jacobian at solution

By default, The finite difference method (`Calculus.jacobian(f)`) is used to approximate the Jacobian. Alternatively, a function which calculates the Jacobian exactly can be supplied and it can be faster and/or more accurate to supply your own jacobian.

```Julia
function jacobian_model(x,p)
    J = Array{Float64}(length(x),length(p))
    J[:,1] = exp.(p[2] * x)    #dmodel/dp[1]
    J[:,2] = x.*p[1].*J[:,1]   #dmodel/dp[2]
    J
end

fit = curve_fit(model, jacobian_model, xdata, ydata, p0)
```

Weight vector can also be assigned to different samples (residuals). `LsqFit.jl` will return the parameters that minimize the weighted residual.

!!! note

  Weight vector is optional. Although unweighted fitting is the numerical equivalent of `w=1.0` for each point, they are handled differently and it is recommended not to provide weight vector of ones, e.g. `[1., 1., 1.]`.

```Julia
fit = curve_fit(model, jacobian_model, xdata, ydata, [0.5, 0.5, 1.], p0)
```

## Linear Approximation
Since the data ``\mathrm{X}`` is given, we can rewrite the function ``h(\mathrm{X}; \theta)`` as ``\eta(\theta)``.

```math
Y = \eta(\theta) + \epsilon
```

There's also a solution parameter vector ``\theta^*``. When ``\theta`` is close to ``\theta^*``, the function ``\eta(\theta)`` can be approximated using Taylor series as

```math
\eta(\theta) \approx \eta(\theta^*) + \mathrm{J}\langle\theta^*\rangle (\theta-\theta^*)
```

where ``\mathrm{J}\langle\theta^*\rangle`` is the Jacobian at the solution parameter. The model can now be tranformed to

```math
Y = \eta(\theta^*) + \mathrm{J}\langle\theta^*\rangle (\theta-\theta^*)+\epsilon
```

and we get a linear model

```math
\widetilde{Y} = \mathrm{J}\langle\theta^*\rangle\beta+\epsilon
```

where the residual is ``Y - \eta(\theta^*) = \widetilde{Y}``, the regressor is ``\mathrm{J}\langle\theta^*\rangle`` and the coefficient is ``\beta = (\theta-\theta^*)``. This sets the foundation for assessing the goodness of fit.

## Goodness of Fit
After getting the estimated parameters ``\hat{\theta}``, we ask the question: how reliable are them? From the linear approximation and assuming errors are normal distributed, the covariance matrix follows the linear model and is therefore

```math
\mathrm{Cov}(\hat{\theta}) = \hat{\sigma}^2(\mathrm{J}\langle\theta^*\rangle^T\,\mathrm{J}\langle\theta^*\rangle)^{-1}
```

and the weighted version, assuming the weight is the inverse covariance of observations, is

```math
\mathrm{Cov}(\hat{\theta}) = \hat{\sigma}^2(\mathrm{J}\langle\theta^*\rangle^T\,\mathrm{w}\,\mathrm{J}\langle\theta^*\rangle)^{-1}
```

In `LsqFit.jl`, the unweighted version of covariance matrix uses QR decomposition to [be more computationally stable](http://www.seas.ucla.edu/~vandenbe/133A/lectures/ls.pdf), which has the form

```math
\mathrm{Cov}(\hat{\theta}) = \hat{\sigma}^2 \mathrm{R}^{-1}(\mathrm{R}^{-1})^T
```

`estimate_covar()` computes the covariance matrix of fit.

```Julia
julia> cov = estimate_covar(fit)
2×2 Array{Float64,2}:
 0.000116545  0.000174633
 0.000174633  0.00258261
```

The standard error is then the square root of each diagonal elements of the covariance matrix. `standard_error` returns the standard error of each parameter.

```Julia
julia> se = standard_error(fit)
2-element Array{Float64,1}:
 0.0114802
 0.0520416
```

`margin_error()` computes the product of standard error and the critical value of each parameter at certain significance level (default is 5%) from t-distribution. The margin of error at 10% significance level can be computed by running

```Julia
julia> margin_of_error = margin_error(fit, 0.1)
2-element Array{Float64,1}:
 0.0199073
 0.0902435
```

`confidence_interval()` returns the confidence interval of each parameter at certain significance level, which is essentially the estimate value ± margin of error. To get the confidence interval at 10% significance level, run

```Julia
julia> confidence_interval = confidence_interval(fit, 0.1)
2-element Array{Tuple{Float64,Float64},1}:
 (0.976316, 1.01613)
 (1.91047, 2.09096)
```

## Reference
Ruckstuhl, A. ‘Introduction to Nonlinear Regression’, Nonlinear Regression, p. 30.
