
# Getting Started

First, import the package.

```julia
julia> using LsqFit
```

Define a two-parameter exponential model, where ``x_t`` is output, ``x_0`` and ``k`` are parameters.

```math
x_t = x_0 e^{kx}
```

```julia
julia> # x: array of independent variables
julia> # p: array of model parameters
julia> model(x, p) = p[1] * exp.(p[2] * x)
```

For illustration purpose, we generate some fake data.

```julia
julia> # xdata: independent variables
julia> # ydata: dependent variable
julia> xdata = linspace(0,10,20)
julia> ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
```

Before fitting the data, we also need define a initial value of parameters for `curve_fit()`.

```julia
julia> p0 = [0.5, 0.5]
```

Run `curve_fit()` to estimate parameters.

```julia
julia> fit = curve_fit(model, xdata, ydata, p0)
```

It will return a composite type (`LsqFitResult`), with some interesting values:

-	`fit.dof`: degrees of freedom
-	`fit.param`: best fit parameters
-	`fit.resid`: residuals = vector of residuals
-	`fit.jacobian`: estimated Jacobian at solution

There are also several functions to help estimate the error. `standard_error` returns the standard error of each parameter.

```Julia
julia> sigma = standard_error(fit)
2-element Array{Float64,1}:
 0.0114802
 0.0520416
```

`margin_error()` computes the product of standard error and critical value of each parameter at certain significance level (default is 5%). The margin of error at 10% significance level can be computed by running

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

The finite difference method is used above to approximate the Jacobian. Alternatively, a function which calculates the Jacobian exactly can be supplied instead.

```Julia
function jacobian_model(x,p)
    J = Array{Float64}(length(x),length(p))
    J[:,1] = exp.(p[2] * x)    #dmodel/dp[1]
    J[:,2] = x.*p[1].*J[:,1]   #dmodel/dp[2]
    J
end

fit = curve_fit(model, jacobian_model, xdata, ydata, p0)
```

For more details of `LsqFit.jl`, check the API section.
