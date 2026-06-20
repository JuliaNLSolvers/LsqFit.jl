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

with `σ²` estimated from the residuals as `σ̂² = `RSS`/(n-p)` ([`LsqFit.mse`](@ref)).
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
| `1 ./ σ.^2` (`Vector`) | known | `(JᵀWJ)⁻¹` | `n` |
| `inv(Σ)` (`Matrix`) | known | `(JᵀWJ)⁻¹` | `n` |
| [`AnalyticWeights`](@ref)`(1 ./ σ.^2)` | estimated | `σ̂² (JᵀWJ)⁻¹` | `n` |
| [`FrequencyWeights`](@ref)`(counts)` | estimated | `σ̂² (JᵀWJ)⁻¹` | `∑w` |

The typed weights are re-exported from
[StatsBase](https://juliastats.org/StatsBase.jl/stable/weights/) and keep their
StatsBase meaning:

* `AnalyticWeights` are reliability or inverse-variance weights. They give a
  relative importance, so the common scale is estimated and the result does not
  depend on the overall magnitude of the weights. This is the convention used by
  MATLAB `nlinfit`, Origin and LabPlot.
* `FrequencyWeights` are integer counts: `wᵢ` means observation `i` was seen `wᵢ`
  times, so `nobs = ∑w`.
* A bare vector or matrix is taken as the exact inverse (co)variance, so the
  scale is known.

`ProbabilityWeights` would require a survey/sandwich variance, and a generic
`Weights` has no associated bias correction in StatsBase. Passing either throws
an error rather than returning a covariance that does not match the type.

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

Taking the weights as exact inverse variances (bare vector, known scale):

```@example weights
fit_known = curve_fit(model, x, y, wt, [2.0, 0.3])
coef(fit_known), stderror(fit_known)
```

Taking them as relative precisions and estimating the scale:

```@example weights
fit_rel = curve_fit(model, x, y, AnalyticWeights(wt), [2.0, 0.3])
coef(fit_rel), stderror(fit_rel)
```

The point estimates agree (`A ≈ 2.263`, `B ≈ 0.275`); only the standard errors
differ. The bare-vector errors are about `±0.097` / `±0.009`, the
`AnalyticWeights` errors about `±0.258` / `±0.024`. The latter match the values
reported by Origin, LabPlot and mycurvefit for this data.

`AnalyticWeights` do not depend on the overall scale of the weights:

```@example weights
se1 = stderror(curve_fit(model, x, y, AnalyticWeights(wt), [2.0, 0.3]))
se2 = stderror(curve_fit(model, x, y, AnalyticWeights(10 .* wt), [2.0, 0.3]))
se1 ≈ se2
```

A bare vector does depend on the scale, and `AnalyticWeights(ones(n))` reproduces
the unweighted fit while a bare vector of ones does not.

For repeated measurements use `FrequencyWeights`; `nobs` is then the total count:

```@example weights
fit_freq = curve_fit(model, x, y, FrequencyWeights([3, 1, 1, 2, 1]), [2.0, 0.3])
nobs(fit_freq), dof(fit_freq)
```

## Monte-Carlo coverage

A 95 % confidence interval should contain the true parameter about 95 % of the
time. The following simulates from a known model with known heteroscedastic
noise and counts how often [`confint`](@ref) covers the truth.

```@example weights
using Random

truth = [2.0, 0.25]
xg = collect(range(0.5, 6; length = 15))
rng = MersenneTwister(20240617)
N = 4000

covers(lohi, t) = [lo <= t <= hi for ((lo, hi), t) in zip(lohi, (t[1], t[2]))]

hitK = zeros(2); hitA = zeros(2); hitU = zeros(2)
for _ in 1:N
    sd = 0.10 .* model(xg, truth)
    yg = model(xg, truth) .+ sd .* randn(rng, length(xg))
    w  = 1 ./ sd.^2
    fK = curve_fit(model, xg, yg, w, copy(truth))                   # known variance
    fA = curve_fit(model, xg, yg, AnalyticWeights(w), copy(truth))  # estimate scale
    fU = curve_fit(model, xg, yg, copy(truth))                      # unweighted
    global hitK += covers(confint(fK; level = 0.95), truth)
    global hitA += covers(confint(fA; level = 0.95), truth)
    global hitU += covers(confint(fU; level = 0.95), truth)
end
(known = round.(100 .* hitK ./ N; digits = 1),
 analytic = round.(100 .* hitA ./ N; digits = 1),
 unweighted = round.(100 .* hitU ./ N; digits = 1))
```

At `N = 30 000` the coverages are about:

| 95 % CIs | A | B |
|:---------|:--|:--|
| bare vector (known variance) | 97 % | 97 % |
| `AnalyticWeights` (estimate scale) | 95 % | 95 % |
| unweighted (ignores heteroscedasticity) | mis-calibrated | mis-calibrated |

`AnalyticWeights` estimate the scale, so `(γ̂ₖ − γₖ)/seₖ` follows Student-t with
`dof = n − p` and the interval is calibrated. With a known variance the correct
multiplier is the normal `z = 1.96`, but for backwards compatibility `confint`
keeps the Student-t reference (`t(13) = 2.16`) for a bare vector, which widens
the interval by about 10 % and raises coverage to 97 %. The standard errors are
unchanged; using the normal quantile recovers 95 %. Ignoring known
heteroscedasticity is the case that is actually wrong.

`margin_error` and `confint` use Student-t for the unweighted case and for
untyped inputs (a bare vector or a matrix). Typed `AbstractWeights` use Student-t
when the scale is estimated and the normal quantile when it is known.

## Choosing weights

* Known `σᵢ`, trusted as absolute: bare vector `1 ./ σ.^2`, or `inv(Σ)` for
  correlated errors.
* Only relative precisions known, or results that match MATLAB/Origin/LabPlot:
  `AnalyticWeights(1 ./ σ.^2)`.
* Repeated or aggregated counts: `FrequencyWeights(counts)`.
* Nothing better: leave `wt` out.
