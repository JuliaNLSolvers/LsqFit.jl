
# Getting Started

First, import the package.

```julia-repl
julia> using LsqFit
```

Define a two-parameter exponential decay model, where ``t`` is a one-element independent variable, ``p_1`` and ``p_2`` are parameters.

The model function is:

```math
m(t, \boldsymbol{p}) = p_1 \exp(-p_2 t)
```

```julia
# t: array of independent variable
# p: array of model parameters
model(t, p) = p[1] * exp.(-p[2] * t)
```

For illustration purpose, we generate some fake data.

```julia
# tdata: data of independent variable
tdata = range(0, stop=10, length=20)
# ydata: data of dependent variable
ydata = model(tdata, [1.0 2.0]) + 0.01*randn(length(tdata))
```

Before fitting the data, we also need a initial value of parameters for `curve_fit()`.

```julia
p0 = [0.5, 0.5]
```

Run `curve_fit()` to fit the data and get the estimated parameters.

```julia-repl
julia> fit = curve_fit(model, tdata, ydata, p0)
julia> param = fit.param
2-element Vector{Float64}:
 1.01105
 2.0735
```

`LsqFit.jl` also provides functions to examine the goodness of fit. `vcov(fit)` computes the estimated covariance matrix.

```julia-repl
julia> cov = vcov(fit)
2×2 Matrix{Float64}:
 0.000116545  0.000174633
 0.000174633  0.00258261
```

`stderror(fit)` returns the standard error of each parameter.

```julia-repl
julia> se = stderror(fit)
2-element Vector{Float64}:
 0.0107956
 0.0508193
```

To get the confidence interval at 10% significance level, run `confint(fit; level=0.9)`, which essentially computes `the estimate parameter value` ± (`standard error` * `critical value from t-distribution`).

```julia-repl
julia> confidence_interval = confint(fit; level=0.9)
2-element Vector{Tuple{Float64,Float64}}:
 (0.992333, 1.02977)
 (1.98537, 2.16162)
```

For more details of `LsqFit.jl`, check [Tutorial](./tutorial.md) and [API References](./api.md) section.
