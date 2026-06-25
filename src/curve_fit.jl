struct LsqFitResult{P,R,J,W<:AbstractArray,T}
    param::P
    resid::R
    jacobian::J
    converged::Bool
    trace::T
    wt::W
end

StatsAPI.coef(lfr::LsqFitResult) = lfr.param
StatsAPI.dof(lfr::LsqFitResult) = nobs(lfr) - length(coef(lfr))
# With FrequencyWeights each weight is a count, so nobs is their sum; otherwise
# it is the number of residuals.
StatsAPI.nobs(lfr::LsqFitResult) = _nobs(lfr.wt, lfr.resid)
_nobs(wt, resid) = length(resid)
_nobs(wt::FrequencyWeights, resid) = sum(wt)
StatsAPI.rss(lfr::LsqFitResult) = sum(abs2, lfr.resid)
StatsAPI.weights(lfr::LsqFitResult) = lfr.wt
StatsAPI.residuals(lfr::LsqFitResult) = lfr.resid
mse(lfr::LsqFitResult) = rss(lfr) / dof(lfr)
isconverged(lsr::LsqFitResult) = lsr.converged

function check_data_health(xdata, ydata, wt = [])
    if any(ismissing, xdata)
        error(
            "The independent variable (`x`) contains `missing` values and a fit cannot be performed",
        )
    end
    if any(ismissing, ydata)
        error(
            "The dependent variable (`y`) contains `missing` values and a fit cannot be performed",
        )
    end
    if any(ismissing, wt)
        error("Weight data contains `missing` values and a fit cannot be performed")
    end
    if any(!isfinite, xdata)
        error(
            "The independent variable (`x`) contains non-finite (e.g. `Inf`, `NaN`) values and a fit cannot be performed",
        )
    end
    if any(!isfinite, ydata)
        error(
            "The dependent variable (`y`) contains non-finite (e.g. `Inf`, `NaN`) values and a fit cannot be performed",
        )
    end
    if any(!isfinite, wt)
        error(
            "Weight contains non-finite (e.g. `Inf`, `NaN`) values and a fit cannot be performed",
        )
    end

end

# provide a method for those who have their own Jacobian function
function lmfit(f, g, p0::AbstractArray, wt::AbstractArray; kwargs...)
    r = f(p0)
    R = OnceDifferentiable(f, g, p0, copy(r); inplace = false)
    lmfit(R, p0, wt; kwargs...)
end

# for inplace f and inplace g
function lmfit(f!, g!, p0::AbstractArray, wt::AbstractArray, r::AbstractArray; kwargs...)
    R = OnceDifferentiable(f!, g!, p0, copy(r); inplace = true)
    lmfit(R, p0, wt; kwargs...)
end

# for inplace f only
function lmfit(
    f,
    p0::AbstractArray,
    wt::AbstractArray,
    r::AbstractArray;
    # autodiff = AutoFiniteDiff(fdjtype = Val(:central)),
    autodiff = AutoForwardDiff(),
    kwargs...,
)
    R = OnceDifferentiable(
        f,
        p0,
        copy(r);
        inplace = true,
        autodiff = autodiff,
    )
    lmfit(R, p0, wt; kwargs...)
end

function lmfit(
    f,
    p0::AbstractArray,
    wt::AbstractArray;
    # autodiff = AutoFiniteDiff(fdjtype = Val(:central)),
    autodiff = AutoForwardDiff(),
    kwargs...,
)
    # this is a convenience function for the curve_fit() methods
    # which assume f(p) is the cost functionj i.e. the residual of a
    # model where
    #   model(xpts, params...) = ydata + error (noise)

    # this minimizes f(p) using a least squares sum of squared error:
    #   rss = sum(f(p)^2)
    #
    # returns p, f(p), g(p) where
    #   p    : best fit parameters
    #   f(p) : function evaluated at best fit p, (weighted) residuals
    #   g(p) : estimated Jacobian at p (Jacobian with respect to p)

    # construct Jacobian function, which uses finite difference method
    r = f(p0)
    R = OnceDifferentiable(
        f,
        p0,
        copy(r);
        inplace = false,
        autodiff = autodiff,
    )
    lmfit(R, p0, wt; kwargs...)
end

function _has_dual_args(e::MethodError)
    return any(e.args) do arg
        T = eltype(typeof(arg))
        T <: ForwardDiff.Dual
    end
end

function lmfit(R::OnceDifferentiable, p0::AbstractArray, wt::AbstractArray; kwargs...)
    results = try
        levenberg_marquardt(R, p0; kwargs...)
    catch e
        if e isa MethodError && _has_dual_args(e)
            throw(
                ArgumentError(
                    """
Model is not compatible with `AutoForwardDiff`: `$(e.f)` has no method accepting \
`ForwardDiff.Dual` numbers.
Either:
  - switch to finite differences: `autodiff = AutoFiniteDiff(fdjtype = Val(:central))`
  - or make the model more permissive w.r.t. the type of the parameters vector, e.g.,
    replace `p::Vector{Float64}` with `p::AbstractVector{<:Real}`""",
                ),
            )
        end
        rethrow(e)
    end
    p = results.minimizer
    converged = isconverged(results)
    return LsqFitResult(p, value!(R, p), jacobian!(R, p), converged, results.trace, wt)
end

# With `scalar = true` the user passes a model `m(xᵢ, p)::Real` that takes a single
# observation, instead of a vectorised `m(xdata, p)::AbstractVector`. We evaluate it
# once per element of `xdata` to build the residual vector the cost function expects.
_scalarized_model(model) = (xdata, p) -> map(x -> model(x, p), xdata)
# A scalar Jacobian `jac(xᵢ, p)::AbstractVector` gives the gradient for one
# observation; stack the per-observation rows into the `nobs × nparams` Jacobian.
_scalarized_jacobian(jac) = (xdata, p) -> mapreduce(x -> permutedims(jac(x, p)), vcat, xdata)

# In-place counterparts (`scalar = true, inplace = true`). The residual is filled
# one observation at a time, and the in-place scalar Jacobian writes each gradient
# straight into its row of `J` via a view, so no per-observation vector is allocated.
function _scalar_residual!(F, model, xdata, p)
    for i in eachindex(F)
        F[i] = model(xdata[i], p)
    end
    return F
end
function _scalar_jacobian!(J, jac!, xdata, p)
    for i in axes(J, 1)
        jac!(view(J, i, :), xdata[i], p)
    end
    return J
end

"""
    curve_fit(model, xdata, ydata, p0) -> fit
    curve_fit(model, xdata, ydata, wt, p0) -> fit

Fit data to a non-linear `model`. `p0` is an initial model parameter guess (see Example),
and `wt` is an optional weighting of the observations.
The return object is a composite type (`LsqFitResult`), with some interesting values:

* `fit.resid` : residuals = vector of residuals
* `fit.jacobian` : estimated Jacobian at solution

additionally, it is possible to query the degrees of freedom with

* `dof(fit)`
* `coef(fit)`

## Scalar models

By default `model(xdata, p)` is vectorised: it receives the whole `xdata` and
returns one prediction per observation. Pass `scalar = true` to instead supply a
model `model(xᵢ, p)` that takes a single observation and returns a single number;
it is then evaluated once per element of `xdata`. An analytic Jacobian, if given,
must likewise be scalar: `jacobian_model(xᵢ, p)` returns the gradient
`∂model/∂p` (a length-`nparams` vector) for that observation.

```julia
fit = curve_fit((x, p) -> p[1] * exp(p[2] * x), xdata, ydata, p0; scalar = true)
```

With `scalar = true, inplace = true` the residual is filled one observation at a
time and the scalar Jacobian writes each gradient straight into its row of `J`
(no per-observation allocation): `jacobian_model(Jᵢ, xᵢ, p)` writes the length-
`nparams` gradient into the row view `Jᵢ`.

```julia
function jac!(Jᵢ, x, p)        # fills row i of the Jacobian in place
    Jᵢ[1] = exp(p[2] * x)
    Jᵢ[2] = p[1] * x * exp(p[2] * x)
end
fit = curve_fit(model, jac!, xdata, ydata, p0; scalar = true, inplace = true)
```

## Weights

The values of `wt` are inverse variances, not standard deviations. A vector
weights the squared residuals `∑ wtᵢ (model−y)ᵢ²`, so the optimal choice is
`wtᵢ = 1/σᵢ²`. A matrix is the inverse covariance `Σ⁻¹` of the observations (a
diagonal of `1/σᵢ²` is the vector case).

The type of `wt` sets whether the residual scale `σ²` is known or estimated:

| `wt` | scale `σ²` | `vcov(fit)` |
|:-----|:-----------|:------------|
| omitted | estimated | `σ̂² (JᵀJ)⁻¹` |
| `PrecisionWeights(1 ./ σ.^2)` | known | `(JᵀWJ)⁻¹` |
| `PrecisionMatrix(inv(Σ))` | known | `(JᵀWJ)⁻¹` |
| `AnalyticWeights(1 ./ σ.^2)` | estimated | `σ̂² (JᵀWJ)⁻¹` |
| `FrequencyWeights(counts)` | estimated, `nobs=∑w` | `σ̂² (JᵀWJ)⁻¹` |

`AnalyticWeights` are relative inverse-variance weights and do not depend on the
overall scale of the weights, matching MATLAB `nlinfit`, Origin and LabPlot.
`PrecisionWeights` are the exact inverse variances and `PrecisionMatrix` the exact
inverse covariance, with the scale known; both use the normal critical value for
confidence intervals. `vcov(fit)` is the parameter covariance and `stderror(fit)`
its square-rooted diagonal. The "Weights" page of the manual has the derivation
and a coverage check.

!!! note
    `wt` must be one of the weight types above. Passing a bare inverse-variance
    `Vector` or inverse-covariance `Matrix` is an error (deprecated in 0.16,
    removed in 1.0); wrap it in `PrecisionWeights(1 ./ σ.^2)` /
    `PrecisionMatrix(inv(Σ))` for a known variance, or `AnalyticWeights` to
    estimate the scale.

## Example
```julia
# a two-parameter exponential model
# x: array of independent variables
# p: array of model parameters
model(x, p) = p[1]*exp.(-x.*p[2])

# some example data
# xdata: independent variables
# ydata: dependent variable
xdata = range(0, stop=10, length=20)
ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
p0 = [0.5, 0.5]

fit = curve_fit(model, xdata, ydata, p0)
```
"""
function curve_fit end

function curve_fit(
    model,
    xdata::AbstractArray,
    ydata::AbstractArray,
    p0::AbstractArray;
    inplace = false,
    scalar = false,
    kwargs...,
)
    check_data_health(xdata, ydata)
    # construct the cost function
    T = eltype(ydata)

    if inplace
        f! =
            scalar ?
            (F, p) -> (_scalar_residual!(F, model, xdata, p); @. F = F - ydata) :
            (F, p) -> (model(F, xdata, p); @. F = F - ydata)
        lmfit(f!, p0, T[], ydata; kwargs...)
    else
        m = scalar ? _scalarized_model(model) : model
        f = (p) -> m(xdata, p) - ydata
        lmfit(f, p0, T[]; kwargs...)
    end
end

function curve_fit(
    model,
    jacobian_model,
    xdata::AbstractArray,
    ydata::AbstractArray,
    p0::AbstractArray;
    inplace = false,
    scalar = false,
    kwargs...,
)
    check_data_health(xdata, ydata)

    T = eltype(ydata)

    if inplace
        f! =
            scalar ?
            (F, p) -> (_scalar_residual!(F, model, xdata, p); @. F = F - ydata) :
            (F, p) -> (model(F, xdata, p); @. F = F - ydata)
        g! =
            scalar ? (G, p) -> _scalar_jacobian!(G, jacobian_model, xdata, p) :
            (G, p) -> jacobian_model(G, xdata, p)
        lmfit(f!, g!, p0, T[], copy(ydata); kwargs...)
    else
        m = scalar ? _scalarized_model(model) : model
        jm = scalar ? _scalarized_jacobian(jacobian_model) : jacobian_model
        f = (p) -> m(xdata, p) - ydata
        g = (p) -> jm(xdata, p)
        lmfit(f, g, p0, T[]; kwargs...)
    end
end

function curve_fit(
    model,
    xdata::AbstractArray,
    ydata::AbstractArray,
    wt::AbstractArray,
    p0::AbstractArray;
    inplace = false,
    scalar = false,
    kwargs...,
)
    check_data_health(xdata, ydata, wt)
    _assert_typed_weights(wt)
    # construct a weighted cost function, with a vector weight for each ydata
    # for example, this might be wt = 1/sigma where sigma is some error term
    u = sqrt.(wt) # to be consistant with the matrix form

    if inplace
        f! =
            scalar ?
            (F, p) -> (_scalar_residual!(F, model, xdata, p); @. F = u * (F - ydata)) :
            (F, p) -> (model(F, xdata, p); @. F = u * (F - ydata))
        lmfit(f!, p0, wt, ydata; kwargs...)
    else
        m = scalar ? _scalarized_model(model) : model
        f = (p) -> u .* (m(xdata, p) - ydata)
        lmfit(f, p0, wt; kwargs...)
    end
end

function curve_fit(
    model,
    jacobian_model,
    xdata::AbstractArray,
    ydata::AbstractArray,
    wt::AbstractArray,
    p0::AbstractArray;
    inplace = false,
    scalar = false,
    kwargs...,
)
    check_data_health(xdata, ydata, wt)
    _assert_typed_weights(wt)
    u = sqrt.(wt) # to be consistant with the matrix form

    if inplace
        f! =
            scalar ?
            (F, p) -> (_scalar_residual!(F, model, xdata, p); @. F = u * (F - ydata)) :
            (F, p) -> (model(F, xdata, p); @. F = u * (F - ydata))
        g! =
            scalar ?
            (G, p) -> (_scalar_jacobian!(G, jacobian_model, xdata, p); @. G = u * G) :
            (G, p) -> (jacobian_model(G, xdata, p); @. G = u * G)
        lmfit(f!, g!, p0, wt, ydata; kwargs...)
    else
        m = scalar ? _scalarized_model(model) : model
        jm = scalar ? _scalarized_jacobian(jacobian_model) : jacobian_model
        f = (p) -> u .* (m(xdata, p) - ydata)
        g = (p) -> u .* (jm(xdata, p))
        lmfit(f, g, p0, wt; kwargs...)
    end
end

function curve_fit(
    model,
    xdata::AbstractArray,
    ydata::AbstractArray,
    wt::AbstractMatrix,
    p0::AbstractArray;
    scalar = false,
    kwargs...,
)
    check_data_health(xdata, ydata, wt)
    _assert_typed_weights(wt)

    # as before, construct a weighted cost function with where this
    # method uses a matrix weight.
    # for example: an inverse_covariance matrix

    # Cholesky is effectively a sqrt of a matrix, which is what we want
    # to minimize in the least-squares of levenberg_marquardt()
    # This requires the matrix to be positive definite
    u = cholesky(wt).U

    m = scalar ? _scalarized_model(model) : model
    f(p) = u * (m(xdata, p) - ydata)
    lmfit(f, p0, wt; kwargs...)
end

function curve_fit(
    model,
    jacobian_model,
    xdata::AbstractArray,
    ydata::AbstractArray,
    wt::AbstractMatrix,
    p0::AbstractArray;
    scalar = false,
    kwargs...,
)
    check_data_health(xdata, ydata, wt)
    _assert_typed_weights(wt)

    u = cholesky(wt).U

    m = scalar ? _scalarized_model(model) : model
    jm = scalar ? _scalarized_jacobian(jacobian_model) : jacobian_model
    f(p) = u * (m(xdata, p) - ydata)
    g(p) = u * (jm(xdata, p))
    lmfit(f, g, p0, wt; kwargs...)
end

"""
    PrecisionWeights(vs)

Weights whose values are the exact, known inverse variances `1 ./ σ.^2` of the
observations. The residual scale is treated as known rather than estimated, so
`vcov` is `(JᵀWJ)⁻¹` with no mean-squared-error factor and confidence intervals
use the normal (asymptotic) critical value.

This is the typed equivalent of passing a bare inverse-variance vector. Use
`AnalyticWeights` instead when the weights are only relative precisions and the
scale should be estimated.
"""
struct PrecisionWeights{S<:Real,T<:Real,V<:AbstractVector{T}} <: AbstractWeights{S,T,V}
    values::V
    sum::S
end
function PrecisionWeights(vs::AbstractVector{<:Real})
    s = sum(vs)
    return PrecisionWeights{typeof(s),eltype(vs),typeof(vs)}(vs, s)
end

"""
    PrecisionMatrix(m)

The exact, known **precision matrix** (inverse covariance) `inv(Σ)` of correlated
observations. This is the matrix counterpart of [`PrecisionWeights`](@ref): the
residual scale is treated as known, so `vcov` is `(JᵀWJ)⁻¹` with no
mean-squared-error factor and confidence intervals use the normal critical value.

!!! note
    The argument is the precision matrix `inv(Σ)`, not the covariance `Σ`.
"""
struct PrecisionMatrix{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    values::M
end
Base.size(pm::PrecisionMatrix) = size(pm.values)
Base.getindex(pm::PrecisionMatrix, i::Int, j::Int) = pm.values[i, j]

# Weights must carry their statistical meaning in their type: PrecisionWeights /
# PrecisionMatrix for a known variance, AnalyticWeights / FrequencyWeights for an
# estimated scale. The typed weights subtype `AbstractVector`/`AbstractMatrix`, so
# they reach the same `curve_fit` methods; these dispatches pass them through and
# reject a plain vector or matrix (which was deprecated in 0.16 and removed in 1.0).
_assert_typed_weights(::AbstractWeights) = nothing
_assert_typed_weights(::PrecisionMatrix) = nothing
_assert_typed_weights(::AbstractVector) = throw(ArgumentError("Bare `Vector` weights are no longer supported. Wrap the inverse variances `1 ./ σ.^2` in `PrecisionWeights` (known variance) or `AnalyticWeights` (estimated scale)."))
_assert_typed_weights(::AbstractMatrix) = throw(ArgumentError("Bare `Matrix` weights are no longer supported. Wrap the inverse covariance `inv(Σ)` in `PrecisionMatrix`."))

# Whether the residual scale σ² is estimated from the fit and multiplied into the
# covariance, or assumed known because the weights are the exact inverse
# (co)variance. Unweighted, AnalyticWeights and FrequencyWeights estimate it; a
# bare vector or matrix, PrecisionWeights or PrecisionMatrix, takes it as known.
_estimates_scale(wt::AbstractVector) = isempty(wt)
_estimates_scale(wt::AbstractMatrix) = false
_estimates_scale(wt::AnalyticWeights) = true
_estimates_scale(wt::FrequencyWeights) = true
_estimates_scale(wt::PrecisionWeights) = false
# ProbabilityWeights need a survey/sandwich variance and Weights has no bias
# correction in StatsBase, so error instead of returning a mismatched covariance.
function _estimates_scale(wt::Union{ProbabilityWeights,Weights})
    throw(
        ArgumentError(
            "Covariance estimation is not defined for `$(nameof(typeof(wt)))`. " *
            "Use `AnalyticWeights` (relative inverse-variance weights), " *
            "`FrequencyWeights` (integer counts), a bare vector of exact inverse " *
            "variances `1 ./ σ.^2`, or an inverse-covariance matrix `inv(Σ)`.",
        ),
    )
end

function StatsAPI.vcov(fit::LsqFitResult)
    # Covariance from the QR decomposition of the (weight-folded) Jacobian, which
    # is more stable than inverting JᵀJ directly (inv(JᵀJ) == Rinv * Rinv'). The
    # weights are folded into J, so JᵀJ already equals Jrawᵀ W Jraw.
    J = fit.jacobian
    Q, R = qr(J)
    Rinv = inv(R)
    covar = Rinv * Rinv'

    # Scale by the estimated residual variance unless the weights fix it.
    if _estimates_scale(fit.wt)
        covar = covar * mse(fit)
    end

    return covar
end

function StatsAPI.stderror(fit::LsqFitResult; rtol::Real = NaN, atol::Real = 0)
    # computes standard error of estimates from
    #   fit   : a LsqFitResult from a curve_fit()
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    covar = vcov(fit)
    # then the standard errors are given by the sqrt of the diagonal
    vars = diag(covar)
    vratio = minimum(vars) / maximum(vars)
    if !isapprox(
        vratio,
        0.0,
        atol = atol,
        rtol = isnan(rtol) ? Base.rtoldefault(vratio, 0.0, 0) : rtol,
    ) && vratio < 0.0
        error("Covariance matrix is negative for atol=$atol and rtol=$rtol")
    end
    return sqrt.(abs.(vars))
end

# Reference distribution for confidence intervals and margins of error. Untyped
# inputs (bare vector, matrix) and the unweighted case keep Student-t for
# backwards compatibility. Typed inputs use Student-t when the scale is estimated
# and the normal quantile when it is known.
_ci_dist(fit::LsqFitResult) = _ci_dist(fit.wt, dof(fit))
_ci_dist(wt, dof) = TDist(dof)
_ci_dist(wt::AbstractWeights, dof) = _estimates_scale(wt) ? TDist(dof) : Normal()
_ci_dist(wt::PrecisionMatrix, dof) = Normal()

function margin_error(fit::LsqFitResult, alpha = 0.05; rtol::Real = NaN, atol::Real = 0)
    # computes margin of error at alpha significance level from
    #   fit   : a LsqFitResult from a curve_fit()
    #   alpha : significance level, e.g. alpha=0.05 for 95% confidence
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    std_errors = stderror(fit; rtol = rtol, atol = atol)
    dist = _ci_dist(fit)
    critical_values = eltype(coef(fit))(quantile(dist, Float64(1 - alpha / 2)))
    # scale standard errors by the quantile of the reference distribution
    return std_errors * critical_values
end

function StatsAPI.confint(fit::LsqFitResult; level = 0.95, rtol::Real = NaN, atol::Real = 0)
    # computes confidence intervals at alpha significance level from
    #   fit   : a LsqFitResult from a curve_fit()
    #   level : confidence level
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    std_errors = stderror(fit; rtol = rtol, atol = atol)
    margin_of_errors = margin_error(fit, 1 - level; rtol = rtol, atol = atol)
    return collect(zip(coef(fit) - margin_of_errors, coef(fit) + margin_of_errors))
end
