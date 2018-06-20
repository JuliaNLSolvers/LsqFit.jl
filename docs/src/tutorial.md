# Tutorial

## Introduction to Nonlinear Regression

We assume that, for the $i$th observation, the relationship between independent variable $\mathbf{x_i}=\begin{bmatrix} x_{1i},\, x_{2i},\, \ldots\\ \end{bmatrix}'$ and dependent variable $Y_i$ follows

```math
Y_i = m(\mathbf{x_i}, \boldsymbol{\gamma}) + \epsilon_i
```

where $m$ is a non-linear model function depends on the independent variable $\mathbf{x_i}$ and the parameter $\boldsymbol{\gamma}$. In order to find the parameter $\boldsymbol{\gamma}$ that "best" fit our data, we can choose the parameter ${\boldsymbol{\gamma}}$ which minimizes the sum of squared residuals from our data, i.e. solves the problem:

```math
\underset{\boldsymbol{\gamma}}{\mathrm{min}} \quad s(\boldsymbol{\gamma})= \sum_{i=1}^{n} [m(\mathbf{x_i}, \boldsymbol{\gamma}) - y_i]^2
```

One example of non-linear model is the exponential model, which has only one predictor variable $t$. The model function is:

```math
m(t, \boldsymbol{\gamma}) = \gamma_1 \exp(\gamma_2 t)
```

and the model becomes:

```math
Y_i = \gamma_1 \exp(\gamma_2 t_i) + \epsilon_i
```

Given that the function $m$ is non-linear, there's no analytical solution for the best $\boldsymbol{\gamma}$. We have to use computational tools, which is `LsqFit.jl` in this tutorial, to find the least squares solution. To fit data using `LsqFit.jl`, pass the defined response function, data and initial value to `curve_fit()`. For now, `LsqFit.jl` only supports the Levenberg Marquardt algorithm.

```julia
julia> # t: array of independent variables
julia> # p: array of model parameters
julia> m(t, p) = p[1] * exp.(p[2] * t)
julia> p0 = [0.5, 0.5]
julia> fit = curve_fit(m, tdata, ydata, p0)
```

It will return return a composite type `LsqFitResult`, with some interesting values:

*	`fit.dof`: degrees of freedom
*	`fit.param`: best fit parameters
*	`fit.resid`: vector of residuals
*	`fit.jacobian`: estimated Jacobian at solution

## Jacobian Calculations

The Jacobian $J(\mathbf{x})$ of a vector function $f(\mathbf{x})$ is deﬁned as the matrix with elements:

```math
[J(\mathbf{x})]_{ij} = \frac{\partial f_i(\mathbf{x})}{\partial x_j}
```
The matrix is therefore:

```math
J(\mathbf{x}) = \begin{bmatrix}
                \frac{\partial f_1}{\partial x_1}&\frac{\partial f_1}{\partial x_2}&\dots&\frac{\partial f_1}{\partial x_n}\\
                \frac{\partial f_2}{\partial x_1}&\frac{\partial f_2}{\partial x_2}&\dots&\frac{\partial f_2}{\partial x_n}\\
                \vdots&\vdots&\ddots&\vdots\\
                \frac{\partial f_m}{\partial x_1}&\frac{\partial f_m}{\partial x_2}&\dots&\frac{\partial f_m}{\partial x_n}\\
                \end{bmatrix}
```

The Jacobian of the exponential model function with respect to $\boldsymbol{\gamma}$ is:

```math
J_m(t, \boldsymbol{\gamma}) = \begin{bmatrix}
            \frac{\partial f}{\partial \gamma_1} &
            \frac{\partial f}{\partial \gamma_2} \\
            \end{bmatrix}
          = \begin{bmatrix}
            \exp(\gamma_2 t) &
            t \gamma_1 \exp(\gamma_2 t) \\
            \end{bmatrix}
```

By default, The finite difference method, `Calculus.jacobian()`, is used to approximate the Jacobian for the data fitting algorithm and covariance computation. Alternatively, a function which calculates the Jacobian can be supplied to `curve_fit()` for faster and/or more accurate results.

```Julia
function j_m(t,p)
    J = Array{Float64}(length(t),length(p))
    J[:,1] = exp.(p[2] .* t)       #df/dp[1]
    J[:,2] = t .* p[1] .* J[:,1]   #df/dp[2]
    J
end

fit = curve_fit(f, j_f, tdata, ydata, p0)
```

## Linear Approximation

The non-linear function $m$ can be approximated as a linear function by Talor expansion:

```math
m(\mathbf{x_i}, \boldsymbol{\gamma}+\boldsymbol{h}) \approx m(\mathbf{x_i}, \boldsymbol{\gamma}) +  \nabla m(\mathbf{x_i}, \boldsymbol{\gamma})'\boldsymbol{h}
```

where $\boldsymbol{h}$ is a very small vector.

Consider the residual vector functon $r({\boldsymbol{\gamma}})$ with entries:

```math
r_i({\boldsymbol{\gamma}}) = Y_i - m(\mathbf{x_i}, {\boldsymbol{\gamma}})
```

The non-linear model can therefore be written as:

```math
\begin{align}
Y_i &\approx m(\mathbf{x_i}, \boldsymbol{\gamma}) +  \nabla m(\mathbf{x_i}, \boldsymbol{\gamma})'\boldsymbol{h} + \epsilon_i \\
Y_i - m(\mathbf{x_i}, \boldsymbol{\gamma}) &\approx \nabla m(\mathbf{x_i}, \boldsymbol{\gamma})'\boldsymbol{h} + \epsilon_i\\
r_i({\boldsymbol{\gamma}}) &\approx \nabla m(\mathbf{x_i}, \boldsymbol{\gamma})'\boldsymbol{h} + \epsilon_i
\end{align}
```

in matrix form:

```math
r({\boldsymbol{\gamma}}) \approx J(\boldsymbol{\gamma})\boldsymbol{h} + \epsilon
```

This is essentially the linear regression model $Y=X\beta$ where $X=J(\boldsymbol{\gamma})$, $\beta=\boldsymbol{h}$ and $Y=r({\boldsymbol{\gamma}})$.
The least squares problem becomes an OLS problem:

```math
\underset{\boldsymbol{h}}{\mathrm{min}} \quad [r(\boldsymbol{\gamma}) - J(\boldsymbol{\gamma})\boldsymbol{h}]'[r(\boldsymbol{\gamma}) - J(\boldsymbol{\gamma})\boldsymbol{h}]
```

The OLS result is:

```math
\hat{\beta}=\hat{\boldsymbol{h}}=(X'X)^{-1}X'XY = [J(\boldsymbol{\gamma})'J(\boldsymbol{\gamma})]^{-1}J(\boldsymbol{\gamma})'J(\boldsymbol{\gamma}) r({\boldsymbol{\gamma}})
```

## Goodness of Fit
Consider the local approximation at a least squares solution $\gamma^*$, which is a local minimizer of the non-linear problem:

```math
\gamma^* = \underset{\boldsymbol{\gamma}}{\mathrm{arg\,min}} \ \sum_{i=1}^{n} [m(\mathbf{x_i}, \boldsymbol{\gamma}) - y_i]^2
```

```math
r({\boldsymbol{\gamma^*}}) \approx J(\boldsymbol{\gamma^*})\boldsymbol{h} + \epsilon
```

where $\boldsymbol{\gamma}=\boldsymbol{\gamma^*} + h$. After getting the estimated parameters ``\hat{\theta}``, we ask the question: how reliable are them? Assume the error is normally distributed, $\epsilon \sim N(0, \sigma^2I)$, the covariance matrix from the linear approximation is

```math
\mathrm{Cov}(\theta^*) = \hat{\sigma}^2(\mathrm{J}\langle\theta^*\rangle'\,\mathrm{J}\langle\theta^*\rangle)^{-1}
```

In `LsqFit.jl`, the unweighted covariance matrix calculation uses QR decomposition to [be more computationally stable](http://www.seas.ucla.edu/~vandenbe/133A/lectures/ls.pdf), which has the form

```math
\mathrm{Cov}(\hat{\theta}) = \hat{\sigma}^2 \mathrm{R}^{-1}(\mathrm{R}^{-1})'
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
julia> confidence_intervals = confidence_interval(fit, 0.1)
2-element Array{Tuple{Float64,Float64},1}:
 (0.976316, 1.01613)
 (1.91047, 2.09096)
```

## Weighted Least Squares
Pass the weight parameter to `curve_fit()`, the parameter that minimizes the weighted residual. Weight parameter (``w``) is a vector or a diagonal matrix of weights for each sample.

```math
\mathbf{W} = \begin{pmatrix}
    w_1    & 0      & \cdots & 0\\
    0      & w_2    & \cdots & 0\\
    \vdots & \vdots & \ddots & \vdots\\
    0      & 0    & \cdots & w_n\\
    \end{pmatrix}

```
The problem to be solved then becomes

```math
\underset{\theta}{\mathrm{arg\,min}} \quad S(\theta)= \sum_{i=1}^{n} w_i[h(\mathbf{x_i}; \theta) - y_i]^2
```

or in matrix form

```math
\underset{\theta}{\mathrm{arg\,min}} \quad S(\theta)=h(\mathbf{X}; \theta) - \mathbf{y}]
```

```Julia
fit = curve_fit(model, jacobian_model, xdata, ydata, w, p0)
```

!!! note

  Weight vector is optional. Although unweighted fitting is the numerical equivalent of `w=1.0` for each point, they are handled differently and it is recommended not to provide weight vector of ones, e.g. `[1., 1., 1.]`.

Assume heteroskedastic error $var(ε) = σ^2 \Omega$ where $\Omega$ is a diagonal matrix and the GLS estimator $\hat{\beta} = (J^T \Omega^{-1} J)^{-1}J^T \Omega^{-1} Y$ which has $cov(\hat{\beta}) = σ^2 (J'  \Omega^{-1}  J)^{-1}$. If $w = var(ε)^{-1} = \Omega^{-1} / \sigma^2$, then $cov(\hat{\beta}) = \sigma^2 (J'  \Omega^{-1}  J)^{-1} = (J'  w  J)^{-1}$.


## References
Hansen, P. C., Pereyra, V. and Scherer, G. (2013) ‘Least Squares Data Fitting with Applications’, p. 147-155.

Kutner, M. H. et al. (2005) Applied linear statistical models.
