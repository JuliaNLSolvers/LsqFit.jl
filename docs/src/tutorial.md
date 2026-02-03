# Tutorial

## Introduction to Nonlinear Regression

Assume that, for the ``i``-th observation, the relationship between independent variable ``\mathbf{x}_i = \begin{bmatrix} x_{1i},\, x_{2i},\, ⋯\, x_{pi}\ \end{bmatrix}'`` and dependent variable ``Y_i`` follows:

```math
Y_i = m(\mathbf{x}_i, \boldsymbol{γ}) + ε_i
```

where ``m`` is a non-linear model function depends on the independent variable ``\mathbf{x}_i`` and the parameter vector ``\boldsymbol{γ}``. In order to find the parameter ``\boldsymbol{γ}`` that "best" fit our data, we choose the parameter ``{\boldsymbol{γ}}`` which minimizes the sum of squared residuals from our data, i.e. solves the problem:

```math
\min_{\boldsymbol{γ}} s(\boldsymbol{γ}) = \sum_{i=1}^n \left[m(\mathbf{x}_i, \boldsymbol{γ}) - y_i\right]^2
```

Given that the function ``m`` is non-linear, there's no analytical solution for the best ``\boldsymbol{γ}``. We have to use computational tools, which is `LsqFit.jl` in this tutorial, to find the least squares solution.

One example of non-linear model is the exponential model, which takes a one-element predictor variable ``t``. The model function is:

```math
m(t, \boldsymbol{γ}) = γ_1 \exp(γ_2 t)
```

and the model becomes:

```math
Y_i = γ_1 \exp(γ_2 t_i) + ε_i
```

To fit data using `LsqFit.jl`, pass the defined model function (`m`), data (`tdata` and `ydata`) and the initial parameter value (`p0`) to `curve_fit()`. For now, `LsqFit.jl` only supports the Levenberg Marquardt algorithm.

```julia
# t: array of independent variables
# p: array of model parameters
m(t, p) = p[1] * exp.(p[2] * t)
p0 = [0.5, 0.5]
fit = curve_fit(m, tdata, ydata, p0)
```

It will return a composite type `LsqFitResult`, with some interesting values:

* `dof(fit)`: degrees of freedom
* `coef(fit)`: best fit parameters
* `fit.resid`: vector of residuals
* `fit.jacobian`: estimated Jacobian at the solution

## Jacobian Calculation

The Jacobian ``J_\mathbf{f}(\mathbf{x})`` of a vector function ``\mathbf{f}(\mathbf{x}): \mathbb{R}^m \to \mathbb{R}^n`` is defined as the matrix with elements:

```math
[J_\mathbf{f}(\mathbf{x})]_{ij} = \frac{∂f_i(\mathbf{x})}{∂x_j}
```
The matrix is therefore:

```math
J_f(\mathbf{x}) = \begin{bmatrix}
    \frac{∂f_1}{∂x_1} & \frac{∂f_1}{∂x_2} & ⋯ & \frac{∂f_1}{∂x_m} \\
    \frac{∂f_2}{∂x_1} & \frac{∂f_2}{∂x_2} & ⋯ & \frac{∂f_2}{∂x_m} \\
    ⋮                  & ⋮                 & ⋱ &  ⋮               \\
    \frac{∂f_n}{∂x_1} & \frac{∂f_n}{∂x_2} & ⋯ & \frac{∂f_n}{∂x_m}
\end{bmatrix}
```

The Jacobian of the exponential model function with respect to ``\boldsymbol{γ}`` is:

```math
J_m(t, \boldsymbol{γ})
= \begin{bmatrix} \frac{∂m}{∂γ_1} & \frac{∂m}{∂γ_2} \end{bmatrix}
= \begin{bmatrix} \exp(γ_2 t) & t γ_1 \exp(γ_2 t) \end{bmatrix}
```

By default, the finite differences is used (see [NLSolversBase.jl](https://github.com/JuliaNLSolvers/NLSolversBase.jl) for more information), is used to approximate the Jacobian for the data fitting algorithm and covariance computation. Alternatively, a function which calculates the Jacobian can be supplied to `curve_fit()` for faster and/or more accurate results.

```Julia
function j_m(t,p)
    J = zeros(length(t),length(p))
    J[:,1] = exp.(p[2] .* t)       #df/dp[1]
    J[:,2] = t .* p[1] .* J[:,1]   #df/dp[2]
    J
end

fit = curve_fit(m, j_m, tdata, ydata, p0)
```

## Linear Approximation

The non-linear function ``m`` can be approximated as a linear function by Taylor expansion:

```math
m(\mathbf{x}_i, \boldsymbol{γ}+\boldsymbol{h}) ≈ m(\mathbf{x}_i, \boldsymbol{γ}) +  ∇ m(\mathbf{x}_i, \boldsymbol{γ})' \boldsymbol{h}
```

where ``\boldsymbol{γ}`` is a fixed vector, ``\boldsymbol{h}`` is a very small-valued vector and ``∇ m(\mathbf{x}_i, \boldsymbol{γ})`` is the gradient at ``\mathbf{x}_i``.

Consider the residual vector function ``\boldsymbol{r}(\boldsymbol{γ}) = \begin{bmatrix} r_1(\boldsymbol{γ}) \\ r_2(\boldsymbol{γ}) \\ ⋮ \\ r_n(\boldsymbol{γ}) \end{bmatrix}`` with entries:

```math
r_i(\boldsymbol{γ}) = m(\mathbf{x}_i, \boldsymbol{γ}) - Y_i
```

Each entry's linear approximation can hence be written as:

```math
\begin{align*}
r_i(\boldsymbol{γ}+\boldsymbol{h}) &= m(\mathbf{x}_i, \boldsymbol{γ}+\boldsymbol{h}) - Y_i\\
&≈ m(\mathbf{x}_i, \boldsymbol{γ}) + ∇ m(\mathbf{x}_i, \boldsymbol{γ})' \boldsymbol{h} - Y_i\\
&= r_i(\boldsymbol{γ}) + ∇ m(\mathbf{x}_i, \boldsymbol{γ})' \boldsymbol{h}
\end{align*}
```

Since the ``i``th row of ``J(\boldsymbol{γ})`` equals the transpose of the gradient of ``m(\mathbf{x}_i, \boldsymbol{γ})``, the vector function ``\boldsymbol{r}(\boldsymbol{γ} + \boldsymbol{h})`` can be approximated as:

```math
\boldsymbol{r}(\boldsymbol{γ}+\boldsymbol{h}) ≈ \boldsymbol{r}(\boldsymbol{γ}) + J(\boldsymbol{γ})\boldsymbol{h}
```

which is a linear function on ``\boldsymbol{h}`` since ``\boldsymbol{γ}`` is a fixed vector.

## Goodness of Fit

The linear approximation of the non-linear least squares problem leads to the approximation of the covariance matrix of each parameter, from which we can perform regression analysis.

Consider a least squares solution ``\boldsymbol{γ}^*``, which is a local minimizer of the non-linear problem:

```math
\boldsymbol{γ}^* = \argmin_{\boldsymbol{γ}} \sum_{i=1}^{n} \left[m(\mathbf{x}_i, \boldsymbol{γ}) - y_i\right]^2
```

Set ``\boldsymbol{γ}^*`` as the fixed point in linear approximation, ``\boldsymbol{r}(\boldsymbol{γ}^*) = \boldsymbol{r}`` and ``J(\boldsymbol{γ}^*) = J``. A parameter vector near ``\boldsymbol{γ}^*`` can be expressed as ``\boldsymbol{γ} = \boldsymbol{γ}^* + \boldsymbol{h}``. The local approximation for the least squares problem is:

```math
\min_{\boldsymbol{γ}} s(\boldsymbol{γ}) = s(\boldsymbol{γ}^* + \boldsymbol{h}) ≈ \left(J\boldsymbol{h} + \boldsymbol{r}\right)' \left(J\boldsymbol{h} + \boldsymbol{r}\right)
```

which is essentially the linear least squares problem:

```math
\min_{\boldsymbol{β}} \left(X\boldsymbol{β} - \boldsymbol{Y}\right)' \left(X\boldsymbol{β} - \boldsymbol{Y}\right)
```

where ``X=J``, ``\boldsymbol{β} = \boldsymbol{h}`` and ``\boldsymbol{Y} = -\boldsymbol{r}(\boldsymbol{γ})``. Solve the equation where the partial derivatives equal to ``0``, the analytical solution is:

```math
\hat{\boldsymbol{h}} = \hat{\boldsymbol{γ}} - \boldsymbol{γ}^* ≈ -[J'J]^{-1} J' \boldsymbol{r}
```

The covariance matrix for the analytical solution is:

```math
\mathbf{Cov}(\hat{\boldsymbol{γ}}) = \mathbf{Cov}(\boldsymbol{h}) = [J'J]^{-1}J' \mathbf{E}(\boldsymbol{r}\boldsymbol{r}') J [J'J]^{-1}
```

Note that ``\boldsymbol{r}`` is the residual vector at the best fit point ``\boldsymbol{γ}^*``, with entries ``r_i = Y_i - m(\mathbf{x}_i, \boldsymbol{γ}^*) = ε_i``. ``\hat{\boldsymbol{γ}}`` is very close to ``\boldsymbol{γ}^*`` and therefore can be replaced by ``\boldsymbol{γ}^*``.

```math
\mathbf{Cov}(\boldsymbol{γ}^*) ≈ \mathbf{Cov}(\hat{\boldsymbol{γ}})
```

Assume the errors in each sample are independent, normal distributed with zero mean and same variance, i.e. ``ε \sim N(0, σ^2I)``, the covariance matrix from the linear approximation is therefore:

```math
\mathbf{Cov}(\boldsymbol{γ}^*) = [J'J]^{-1}J'\mathbf{Cov}(ε)J[J'J]^{-1} = σ^2[J'J]^{-1}
```

where ``σ^2`` could be estimated as residual sum of squares divided by degrees of freedom:

```math
\hat{σ}^2=\frac{s(\boldsymbol{γ}^*)}{n-p}
```
In `LsqFit.jl`, the covariance matrix calculation uses QR decomposition to [be more computationally stable](http://www.seas.ucla.edu/~vandenbe/133A/lectures/ls.pdf), which has the form:

```math
\mathbf{Cov}(\boldsymbol{γ}^*) = \hat{σ}^2 \mathrm{R}^{-1}(\mathrm{R}^{-1})'
```

`vcov()` computes the covariance matrix of fit:

```julia-repl
julia> cov = vcov(fit)
2×2 Matrix{Float64}:
 0.000116545  0.000174633
 0.000174633  0.00258261
```

The standard error is then the square root of each diagonal elements of the covariance matrix. `stderror()` returns the standard error of each parameter:

```julia-repl
julia> se = stderror(fit)
2-element Vector{Float64}:
 0.0114802
 0.0520416
```

`margin_error()` computes the product of standard error and the critical value of each parameter at a certain significance level (default is 5%) from t-distribution. The margin of error at 10% significance level can be computed by:

```julia-repl
julia> margin_of_error = margin_error(fit, 0.1)
2-element Vector{Float64}:
 0.0199073
 0.0902435
```

`confint()` returns the confidence interval of each parameter at certain significance level, which is essentially the estimate value ± margin of error. To get the confidence interval at 10% significance level, run:

```julia-repl
julia> confidence_intervals = confint(fit; level=0.9)
2-element Vector{Tuple{Float64,Float64}}:
 (0.976316, 1.01613)
 (1.91047, 2.09096)
```

## Weighted Least Squares

`curve_fit()` also accepts weight parameter (`wt`) to perform Weighted Least Squares and General Least Squares, where the parameter ``\boldsymbol{γ}^*`` minimizes the weighted residual sum of squares.

Weight parameter (`wt`) is an array or a matrix of weights for each sample. To perform Weighted Least Squares, pass the weight array `[w_1, w_2, ..., w_n]` or the weight matrix `W`:

```math
\mathbf{W} = \begin{bmatrix}
    w_1 & 0   & ⋯ & 0 \\
    0   & w_2 & ⋯ & 0 \\
    ⋮   & ⋮   & ⋱ & ⋮ \\
    0   & 0   & ⋯ & w_n
    \end{bmatrix}
```

The weighted least squares problem becomes:

```math
\min_{\boldsymbol{γ}} s(\boldsymbol{γ}) = \sum_{i=1}^{n} w_i \left[m(\mathbf{x}_i, \boldsymbol{γ}) - Y_i\right]^2
```

in matrix form:

```math
\min_{\boldsymbol{γ}} s(\boldsymbol{γ}) = \boldsymbol{r}(\boldsymbol{γ})' W \boldsymbol{r}(\boldsymbol{γ})
```

where ``r(\boldsymbol{γ}) = \begin{bmatrix} r_1(\boldsymbol{γ}) \\ r_2(\boldsymbol{γ}) \\ ⋮\\ r_n(\boldsymbol{γ}) \end{bmatrix}``
is a residual vector function with entries:

```math
r_i(\boldsymbol{γ}) = m(\mathbf{x}_i, \boldsymbol{γ}) - Y_i
```

The algorithm in `LsqFit.jl` will then provide a least squares solution ``\boldsymbol{γ}^*``.

!!! note
    In `LsqFit.jl`, the residual function passed to `levenberg_marquardt()` is in different format, if the weight is a vector:

    ```julia
    r(p) = sqrt.(wt) .* ( model(xpts, p) - ydata )
    lmfit(r, g, p0, wt; kwargs...)
    ```

    ```math
    r_i(\boldsymbol{γ}) = \sqrt{w_i} \cdot [m(\mathbf{x}_i, \boldsymbol{γ}) - Y_i]
    ```

    Cholesky decomposition, which is effectively a sqrt of a matrix, will be performed if the weight is a matrix:

    ```julia
    u = chol(wt)
    r(p) = u * ( model(xpts, p) - ydata )
    lmfit(r, p0, wt; kwargs...)
    ```

    ```math
    r_i(\boldsymbol{γ}) = \sqrt{w_i} \cdot \left[m(\mathbf{x}_i, \boldsymbol{γ}) - Y_i\right]
    ```

    The solution will be the same as the least squares problem mentioned in the tutorial.

Set ``\boldsymbol{r}({\boldsymbol{γ}^*}) = \boldsymbol{r}`` and ``J(\boldsymbol{γ}^*) = J``, the linear approximation of the weighted least squares problem is then:

```math
\min_{\boldsymbol{γ}} s(\boldsymbol{γ}) = s(\boldsymbol{γ}^* + \boldsymbol{h})
≈ [J\boldsymbol{h}+\boldsymbol{r}]' W [J\boldsymbol{h}+\boldsymbol{r}]
```

The analytical solution to the linear approximation is:

```math
\hat{\boldsymbol{h}} = \hat{\boldsymbol{γ}} - \boldsymbol{γ}^* ≈ -[J'WJ]^{-1} J' W \boldsymbol{r}
```

Assume the errors in each sample are independent, normal distributed with zero mean and **different** variances (heteroskedastic error), i.e. ``ε \sim N(0, \Sigma)``, where:

```math
\Sigma = \begin{bmatrix}
         σ_1^2  & 0     & ⋯  & 0\\
         0      & σ_2^2 & ⋯  & 0\\
         ⋮      & ⋮     & ⋱   & ⋮  \\
         0      & 0     & ⋯  & σ_n^2\\
         \end{bmatrix}
```

We know the error variance and we set the weight as the inverse of the variance (the optimal weight), i.e. ``W = \Sigma^{-1}``:

```math
\mathbf{W} =  \begin{bmatrix}
              w_1 & 0   & ⋯ & 0 \\
              0   & w_2 & ⋯ & 0 \\
              ⋮  & ⋮    & ⋱ & ⋮ \\
              0  & 0    & ⋯ & w_n
              \end{bmatrix}
           =  \begin{bmatrix}
               \frac{1}{σ_1^2} & 0               & ⋯ & 0 \\
               0               & \frac{1}{σ_2^2} & ⋯ & 0 \\
               ⋮               & ⋮               & ⋱ & ⋮ \\
               0               & 0               & ⋯ & \frac{1}{σ_n^2}
               \end{bmatrix}
```

The covariance matrix is now:

```math
\mathbf{Cov}(\boldsymbol{γ}^*) ≈ [J'WJ]^{-1}J'W \Sigma W'J[J'W'J]^{-1} = [J'WJ]^{-1}
```


If we only know **the relative ratio of different variances**, i.e. ``ε \sim N(0, σ^2 W^{-1})``, the covariance matrix will be:

```math
\mathbf{Cov}(\boldsymbol{γ}^*) = σ^2[J'WJ]^{-1}
```

where ``σ^2`` is estimated. In this case, if we set ``W = I``, the result will be the same as the unweighted version. However, `curve_fit()` currently **does not support** this implementation. `curve_fit()` assumes the weight as the inverse of **the error covariance matrix** rather than **the ratio of error covariance matrix**, i.e. the covariance of the estimated parameter is calculated as `covar = inv(J'*fit.wt*J)`.

!!! note
    Passing vector of ones as the weight vector will cause mistakes in covariance estimation.

Pass the vector of `1 ./ var(ε)` or the matrix `inv(covar(ε))` as the weight parameter (`wt`) to the function `curve_fit()`:

```Julia
wt = inv(cov_ε)
fit = curve_fit(m, tdata, ydata, wt, p0)
cov = vcov(fit)
```

!!! note
    If the weight matrix is not a diagonal matrix, General Least Squares will be performed.

## General Least Squares
Assume the errors in each sample are **correlated**, normal distributed with zero mean and **different** variances (heteroskedastic and autocorrelated error), i.e. ``ε \sim N(0, \Sigma)``.

Set the weight matrix as the inverse of the error covariance matrix (the optimal weight), i.e. ``W = \Sigma^{-1}``, we will get the parameter covariance matrix:

```math
\mathbf{Cov}(\boldsymbol{γ}^*) ≈ [J'WJ]^{-1}J'W \Sigma W'J[J'W'J]^{-1} = [J'WJ]^{-1}
```

Pass the matrix `inv(covar(ε))` as the weight parameter (`wt`) to the function `curve_fit()`:

```Julia
wt = 1 ./ yvar
fit = curve_fit(m, tdata, ydata, wt, p0)
cov = vcov(fit)
```

## Estimate the Optimal Weight
In most cases, the variances of errors are unknown. To perform Weighted Least Square, we need estimate the variances of errors first, which is the squared residual of ``i``-th sample:

```math
\widehat{\mathbf{Var}(ε_i)} = \widehat{\mathbf{E}(ε_i ε_i)} = r_i(\boldsymbol{γ}^*)
```

Unweighted fitting (OLS) will return the residuals we need, since the estimator of OLS is unbiased. Then pass the reciprocal of the residuals as the estimated optimal weight to perform Weighted Least Squares:

```Julia
fit_OLS = curve_fit(m, tdata, ydata, p0)
wt = 1 ./ fit_OLS.resid
fit_WLS = curve_fit(m, tdata, ydata, wt, p0)
cov = vcov(fit_WLS)
```

## References
Hansen, P. C., Pereyra, V. and Scherer, G. (2013) Least squares data fitting with applications. Baltimore, Md: Johns Hopkins University Press, p. 147-155.

Kutner, M. H. et al. (2005) Applied Linear statistical models.

Weisberg, S. (2014) Applied linear regression. Fourth edition. Hoboken, NJ: Wiley (Wiley series in probability and statistics).
