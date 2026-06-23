# Weights

This page describes how weighted fits affect the reported parameter
uncertainties, how the weight values and weight types are interpreted, and a
Monte-Carlo check of the resulting confidence intervals.

Two points cover most of the confusion around weighted fits:

* The weight *values* are inverse variances (`wᵢ = 1/σᵢ²`), not standard
  deviations.
* The weight *type* decides whether the noise scale `σ²` is treated as known or
  estimated. `vcov(fit)` returns a parameter covariance matrix and
  `stderror(fit)` the square root of its diagonal.

## Standard deviation, variance and covariance

| Quantity | Symbol | Units | Role |
|:---------|:-------|:------|:-----|
| Observation standard deviation | `σᵢ` | like `yᵢ` | not passed directly |
| Observation variance | `σᵢ²` | `yᵢ²` | a vector weight is `1/σᵢ²` |
| Observation covariance | `Σ` | `yᵢ yⱼ` | a matrix weight is `Σ⁻¹` |
| Parameter covariance | `Cov(γ̂)` | `pᵢ pⱼ` | `vcov(fit)` |
| Parameter standard error | `√Cov(γ̂)ₖₖ` | like `pₖ` | `stderror(fit)` |

The data uncertainties go in as inverse variances and the parameter
uncertainties come out as a covariance matrix whose square-rooted diagonal gives
the standard errors.

If observation `i` has standard deviation `σᵢ`, its weight is `1/σᵢ²`, not
`1/σᵢ`. The weight multiplies the squared residual `∑ wᵢ (model−y)ᵢ²`, and the
minimum-variance weighting of a squared residual is the inverse of its variance.

## Weighted least squares

Near the optimum the weighted problem linearises to

```math
\hat{\boldsymbol{h}} = \hat{\boldsymbol{γ}} - \boldsymbol{γ}^*
    \approx -[J'WJ]^{-1} J' W \boldsymbol{r},
```

with `J` the Jacobian and `W` the weight matrix. For errors `ε ∼ N(0, Σ)` and
the optimal weights `W = Σ⁻¹`,

```math
\mathbf{Cov}(\boldsymbol{γ}^*)
    = [J'WJ]^{-1} J'W\,\Sigma\,W'J\,[J'W'J]^{-1}
    = [J'WJ]^{-1}.
```

This is the known-variance case: the exact `Σ⁻¹` was supplied, so there is no
scale left to estimate.

If the variances are known only up to a common unknown factor,
`ε ∼ N(0, σ² W⁻¹)`, then

```math
\mathbf{Cov}(\boldsymbol{γ}^*) = σ^2 [J'WJ]^{-1},
```

with `σ²` estimated from the residuals as the mean squared error `RSS/(n-p)`
(`LsqFit.mse`).
The same `σ²` multiplies the whole matrix, so this estimator does not change if
all weights are multiplied by a constant. The unweighted fit is the case `W = I`
of this second formula.

Both formulas are correct; they answer different questions.

## Weight types

LsqFit selects the formula from the type of `wt`. The values are the same
inverse variances in every case.

| `wt` argument | scale `σ²` | `vcov(fit)` | `nobs` |
|:--------------|:-----------|:------------|:-------|
| omitted | estimated | `σ̂² (JᵀJ)⁻¹` | `n` |
| [`PrecisionWeights`](@ref)`(1 ./ σ.^2)` | known | `(JᵀWJ)⁻¹` | `n` |
| [`PrecisionMatrix`](@ref)`(inv(Σ))` | known | `(JᵀWJ)⁻¹` | `n` |
| `AnalyticWeights(1 ./ σ.^2)` | estimated | `σ̂² (JᵀWJ)⁻¹` | `n` |
| `FrequencyWeights(counts)` | estimated | `σ̂² (JᵀWJ)⁻¹` | `∑w` |

!!! warning "Deprecated: bare vector and matrix weights"
    Passing a bare `1 ./ σ.^2` vector or `inv(Σ)` matrix is deprecated and emits a
    warning. Wrap them in [`PrecisionWeights`](@ref) / [`PrecisionMatrix`](@ref)
    for the same known-variance covariance (with the calibrated normal interval),
    or in `AnalyticWeights` to estimate the scale. The bare forms previously kept a
    Student-t interval purely for backwards compatibility.

`AnalyticWeights`, `FrequencyWeights` are re-exported from
[StatsBase](https://juliastats.org/StatsBase.jl/stable/weights/) and keep their
StatsBase meaning; `PrecisionWeights` and `PrecisionMatrix` are defined by LsqFit.

* `PrecisionWeights` are the exact, known inverse variances (independent errors)
  and `PrecisionMatrix` the exact, known precision matrix `inv(Σ)` (correlated
  errors). The scale is known, so `vcov` has no MSE factor and the confidence
  interval uses the normal critical value. They are the typed forms of a bare
  vector and a bare matrix.
* `AnalyticWeights` are reliability or inverse-variance weights. They give a
  relative importance, so the common scale is estimated and the result does not
  depend on the overall magnitude of the weights. This is the convention used by
  MATLAB `nlinfit`, Origin and LabPlot.
* `FrequencyWeights` are integer counts: `wᵢ` means observation `i` was seen `wᵢ`
  times, so `nobs = ∑w`.
* A bare vector or matrix (deprecated) is taken as the exact inverse (co)variance,
  like `PrecisionWeights`/`PrecisionMatrix`, but keeps the Student-t interval for
  backwards compatibility. Prefer the typed forms.

StatsBase has no weight type for the known-variance case (all of its corrected
weights estimate the scale), which is why `PrecisionWeights`/`PrecisionMatrix`
are provided here. `ProbabilityWeights` would require a survey/sandwich variance,
and a generic `Weights` has no associated bias correction in StatsBase. Passing
either throws an error rather than returning a covariance that does not match the
type.

## Example

The dataset below is the one from issue #255. The model is `f(x) = A·exp(B·x)`
with a 5 % relative measurement error.

```@example weights
using LsqFit

model(x, p) = p[1] .* exp.(p[2] .* x)

x = Float64[1, 2, 4, 5, 8]
y = Float64[3, 4, 6, 11, 20]
σ = 0.05 .* y          # standard deviations
wt = 1 ./ σ.^2         # inverse variances

nothing # hide
```

Taking the weights as exact inverse variances (known scale):

```@example weights
fit_known = curve_fit(model, x, y, PrecisionWeights(wt), [2.0, 0.3])
coef(fit_known), stderror(fit_known)
```

Taking them as relative precisions and estimating the scale:

```@example weights
fit_rel = curve_fit(model, x, y, AnalyticWeights(wt), [2.0, 0.3])
coef(fit_rel), stderror(fit_rel)
```

The point estimates agree (`A ≈ 2.263`, `B ≈ 0.275`); only the standard errors
differ. The known-scale (`PrecisionWeights`) errors are about `±0.097` / `±0.009`,
the `AnalyticWeights` errors about `±0.258` / `±0.024`. The latter match the values
reported by Origin, LabPlot and mycurvefit for this data.

`AnalyticWeights` do not depend on the overall scale of the weights:

```@example weights
se1 = stderror(curve_fit(model, x, y, AnalyticWeights(wt), [2.0, 0.3]))
se2 = stderror(curve_fit(model, x, y, AnalyticWeights(10 .* wt), [2.0, 0.3]))
se1 ≈ se2
```

`PrecisionWeights` do depend on the scale, and `AnalyticWeights(ones(n))`
reproduces the unweighted fit while `PrecisionWeights(ones(n))` does not.

For repeated measurements use `FrequencyWeights`; `nobs` is then the total count:

```@example weights
fit_freq = curve_fit(model, x, y, FrequencyWeights([3, 1, 1, 2, 1]), [2.0, 0.3])
nobs(fit_freq), dof(fit_freq)
```

## Monte-Carlo coverage

A 95 % confidence interval should contain the true parameter about 95 % of the
time. The following simulates from a known model with known heteroscedastic
noise and counts how often `confint` covers the truth.

```@example weights
using Random

truth = [2.0, 0.25]
xg = collect(range(0.5, 6; length = 15))
rng = MersenneTwister(20240617)
N = 4000

covers(lohi, t) = [lo <= t <= hi for ((lo, hi), t) in zip(lohi, (t[1], t[2]))]

hitV = zeros(2); hitA = zeros(2); hitU = zeros(2)
for _ in 1:N
    sd = 0.10 .* model(xg, truth)
    yg = model(xg, truth) .+ sd .* randn(rng, length(xg))
    w  = 1 ./ sd.^2
    fV = curve_fit(model, xg, yg, PrecisionWeights(w), copy(truth))  # known variance, normal CI
    fA = curve_fit(model, xg, yg, AnalyticWeights(w), copy(truth))  # estimated scale
    fU = curve_fit(model, xg, yg, copy(truth))                      # unweighted
    global hitV += covers(confint(fV; level = 0.95), truth)
    global hitA += covers(confint(fA; level = 0.95), truth)
    global hitU += covers(confint(fU; level = 0.95), truth)
end
nothing # hide
```

```@example weights
using SummaryTables

pct(h) = string(round(100 * h / N; digits = 1), " %")
covrow(name, h) = [Cell(name) Cell(pct(h[1])) Cell(pct(h[2]))]
body = vcat(
    permutedims(Cell.(["weighting", "A coverage", "B coverage"]; bold = true)),
    covrow("PrecisionWeights (known σ², normal)", hitV),
    covrow("AnalyticWeights (estimated scale)", hitA),
    covrow("unweighted (ignores heteroscedasticity)", hitU),
)
Table(body; header = 1)
```

`PrecisionWeights` and `AnalyticWeights` are both close to the nominal 95 %, each
with the reference distribution that matches its assumption. `PrecisionWeights`
treats the variance as known, so `(γ̂ₖ − γₖ)/seₖ` is asymptotically normal and
`confint` uses `z = 1.96`. `AnalyticWeights` estimate the scale, so the same
ratio follows Student-t with `dof = n − p`. Ignoring the known heteroscedasticity
is the case that is actually wrong.

`margin_error` and `confint` use Student-t for the unweighted case. Typed weights
use Student-t when the scale is estimated (`AnalyticWeights`, `FrequencyWeights`)
and the normal quantile when it is known (`PrecisionWeights`, `PrecisionMatrix`).
A deprecated bare vector or matrix has the same covariance as `PrecisionWeights` /
`PrecisionMatrix` but keeps the Student-t reference for backwards compatibility.

## Choosing weights

* Known `σᵢ`, trusted as absolute: `PrecisionWeights(1 ./ σ.^2)`, or
  `PrecisionMatrix(inv(Σ))` for correlated errors. (Passing a bare `1 ./ σ.^2`
  vector or `inv(Σ)` matrix is deprecated.)
* Only relative precisions known, or results that match MATLAB/Origin/LabPlot:
  `AnalyticWeights(1 ./ σ.^2)`.
* Repeated or aggregated counts: `FrequencyWeights(counts)`.
* Nothing better: leave `wt` out.
