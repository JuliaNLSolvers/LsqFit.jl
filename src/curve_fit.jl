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
# For `FrequencyWeights` each weight is a repeat count, so the number of
# observations is the sum of the counts; for every other weighting it is the
# number of residuals.
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

# Convert legacy symbol-based autodiff to AbstractADType for NLSolversBase v8 compatibility.
# Symbols were accepted in v7 but are no longer valid in v8.
function _autodiff_adtype(autodiff::AbstractADType)
    autodiff
end
function _autodiff_adtype(autodiff::Symbol)
    Base.depwarn(
        "Passing `autodiff` as a Symbol (e.g. `$(repr(autodiff))`) is deprecated. " *
        "Use an `ADTypes` backend such as `AutoForwardDiff()` or `AutoFiniteDiff(fdjtype = Val(:central))` instead.",
        :curve_fit,
    )
    if autodiff in (:finite, :central)
        return AutoFiniteDiff(fdjtype = Val(:central))
    elseif autodiff == :finiteforward
        return AutoFiniteDiff(fdjtype = Val(:forward))
    elseif autodiff == :finitecomplex
        return AutoFiniteDiff(fdjtype = Val(:complex))
    elseif autodiff in (:forward, :forwarddiff)
        return AutoForwardDiff()
    else
        throw(
            ArgumentError(
                "Unsupported autodiff symbol: $(repr(autodiff)). Use `AutoFiniteDiff()` or `AutoForwardDiff()`.",
            ),
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
        autodiff = _autodiff_adtype(autodiff),
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
        autodiff = _autodiff_adtype(autodiff),
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

## Weights and what they mean

The *values* you pass are always in **inverse-variance** units, never standard
deviations:

* a **vector** `wt` weights the squared residuals, `∑ wtᵢ (model−y)²ᵢ`, so the
  optimal choice is the **inverse variance** `wtᵢ = 1/σᵢ²` (not `1/σᵢ`);
* a **matrix** `wt` is the **inverse covariance** of the observations, `Σ⁻¹`
  (a diagonal matrix of `1/σᵢ²` is exactly the vector case).

What changes the reported uncertainty is the *type* of `wt`, which states
whether the residual scale `σ²` is known or estimated:

| `wt`                                  | residual scale `σ²` | `vcov(fit)`        |
|:--------------------------------------|:--------------------|:-------------------|
| omitted (unweighted)                  | estimated           | `σ̂² (JᵀJ)⁻¹`       |
| bare vector `1 ./ σ.^2`               | known (`≡ 1`)       | `(JᵀWJ)⁻¹`         |
| matrix `inv(Σ)`                       | known (`≡ 1`)       | `(JᵀWJ)⁻¹`         |
| `AnalyticWeights(1 ./ σ.^2)`          | estimated           | `σ̂² (JᵀWJ)⁻¹`      |
| `FrequencyWeights(counts)`            | estimated, `nobs=∑w`| `σ̂² (JᵀWJ)⁻¹`      |

`AnalyticWeights` are *relative* inverse-variance (reliability/precision)
weights and are scale-invariant — this matches the convention used by MATLAB
`nlinfit`, Origin and LabPlot. A bare vector instead asserts that you know the
exact inverse variances. `vcov(fit)` returns the parameter **covariance** matrix
and `stderror(fit)` its diagonal **standard deviations** (`√diag`). See the
"Weights" page of the manual for the full derivation and Monte-Carlo coverage.

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
    kwargs...,
)
    check_data_health(xdata, ydata)
    # construct the cost function
    T = eltype(ydata)

    if inplace
        f! = (F, p) -> (model(F, xdata, p); @. F = F - ydata)
        lmfit(f!, p0, T[], ydata; kwargs...)
    else
        f = (p) -> model(xdata, p) - ydata
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
    kwargs...,
)
    check_data_health(xdata, ydata)

    T = eltype(ydata)

    if inplace
        f! = (F, p) -> (model(F, xdata, p); @. F = F - ydata)
        g! = (G, p) -> jacobian_model(G, xdata, p)
        lmfit(f!, g!, p0, T[], copy(ydata); kwargs...)
    else
        f = (p) -> model(xdata, p) - ydata
        g = (p) -> jacobian_model(xdata, p)
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
    kwargs...,
)
    check_data_health(xdata, ydata, wt)
    # construct a weighted cost function, with a vector weight for each ydata
    # for example, this might be wt = 1/sigma where sigma is some error term
    u = sqrt.(wt) # to be consistant with the matrix form

    if inplace
        f! = (F, p) -> (model(F, xdata, p); @. F = u * (F - ydata))
        lmfit(f!, p0, wt, ydata; kwargs...)
    else
        f = (p) -> u .* (model(xdata, p) - ydata)
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
    kwargs...,
)
    check_data_health(xdata, ydata, wt)
    u = sqrt.(wt) # to be consistant with the matrix form

    if inplace
        f! = (F, p) -> (model(F, xdata, p); @. F = u * (F - ydata))
        g! = (G, p) -> (jacobian_model(G, xdata, p); @. G = u * G)
        lmfit(f!, g!, p0, wt, ydata; kwargs...)
    else
        f = (p) -> u .* (model(xdata, p) - ydata)
        g = (p) -> u .* (jacobian_model(xdata, p))
        lmfit(f, g, p0, wt; kwargs...)
    end
end

function curve_fit(
    model,
    xdata::AbstractArray,
    ydata::AbstractArray,
    wt::AbstractMatrix,
    p0::AbstractArray;
    kwargs...,
)
    check_data_health(xdata, ydata, wt)

    # as before, construct a weighted cost function with where this
    # method uses a matrix weight.
    # for example: an inverse_covariance matrix

    # Cholesky is effectively a sqrt of a matrix, which is what we want
    # to minimize in the least-squares of levenberg_marquardt()
    # This requires the matrix to be positive definite
    u = cholesky(wt).U

    f(p) = u * (model(xdata, p) - ydata)
    lmfit(f, p0, wt; kwargs...)
end

function curve_fit(
    model,
    jacobian_model,
    xdata::AbstractArray,
    ydata::AbstractArray,
    wt::AbstractMatrix,
    p0::AbstractArray;
    kwargs...,
)
    check_data_health(xdata, ydata, wt)

    u = cholesky(wt).U

    f(p) = u * (model(xdata, p) - ydata)
    g(p) = u * (jacobian_model(xdata, p))
    lmfit(f, g, p0, wt; kwargs...)
end

# Whether the overall residual scale (variance) σ² is *estimated* from the fit
# and multiplied into the covariance (`σ̂² · (JᵀWJ)⁻¹`), or assumed *known* and
# equal to one because the supplied weights already are the exact inverse
# (co)variance (`(JᵀWJ)⁻¹`). See the "Weights" section of the manual.
#
#   - unweighted fit ............... estimate  (the classic σ̂² = RSS/(n−p))
#   - bare inverse-variance vector . known     (you asserted the exact 1/σ²ᵢ)
#   - inverse-covariance matrix .... known     (you asserted the exact Σ⁻¹)
#   - AnalyticWeights .............. estimate  (relative precision; scale-invariant)
#   - FrequencyWeights ............. estimate  (repeat counts; nobs = ∑w)
_estimates_scale(wt::AbstractVector) = isempty(wt)
_estimates_scale(wt::AbstractMatrix) = false
_estimates_scale(wt::AnalyticWeights) = true
_estimates_scale(wt::FrequencyWeights) = true
# `ProbabilityWeights` need a sandwich/survey variance and `Weights` carries no
# covariance semantics (StatsBase itself refuses a bias correction for it), so
# we refuse rather than return a silently wrong covariance.
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
    # computes covariance matrix of fit parameters

    # The covariance is built from the QR decomposition of the (weight-folded)
    # Jacobian, which is numerically more stable than forming and inverting
    # JᵀJ directly (note inv(JᵀJ) == inv(RᵀR) == Rinv * Rinv'). Because the
    # weights are folded into the residuals and hence into J, JᵀJ already
    # equals Jrawᵀ W Jraw.
    J = fit.jacobian
    Q, R = qr(J)
    Rinv = inv(R)
    covar = Rinv * Rinv'

    # Multiply by the estimated residual variance only when the scale is not
    # assumed known from the weights (see `_estimates_scale`).
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

# Reference distribution for confidence intervals / margins of error.
#
# Untyped weight inputs (a bare vector, an inverse-covariance matrix) and the
# unweighted case keep the Student-t reference for backwards compatibility, even
# when the scale is known (for which the normal would be the asymptotically
# correct choice — see the "Weights" manual page). This makes those intervals
# mildly conservative but never anti-conservative, and preserves historical
# behaviour.
#
# Typed `AbstractWeights` instead select the asymptotically-correct reference:
# Student-t when the scale is estimated (`AnalyticWeights`, `FrequencyWeights`)
# and the standard normal when it is known. The normal branch is therefore only
# reachable through a typed, known-scale weight.
_ci_dist(fit::LsqFitResult) = _ci_dist(fit.wt, dof(fit))
_ci_dist(wt, dof) = TDist(dof)
_ci_dist(wt::AbstractWeights, dof) = _estimates_scale(wt) ? TDist(dof) : Normal()

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

@deprecate(
    confidence_interval(fit::LsqFitResult, alpha = 0.05; rtol::Real = NaN, atol::Real = 0),
    confint(fit; level = (1 - alpha), rtol = rtol, atol = atol)
)

@deprecate estimate_covar(fit::LsqFitResult) vcov(fit)

@deprecate standard_errors(args...; kwargs...) stderror(args...; kwargs...)

@deprecate estimate_errors(
    fit::LsqFitResult,
    confidence = 0.95;
    rtol::Real = NaN,
    atol::Real = 0,
) margin_error(fit, 1 - confidence; rtol = rtol, atol = atol)
