# Weights

Weighted least squares is where most of the confusion about LsqFit's uncertainty
estimates comes from. This page makes the roles of **standard deviation**,
**variance** and **covariance** explicit, derives the two covariance formulas,
shows them on a worked example, and finally *verifies* them with a Monte-Carlo
coverage study so you can trust the numbers rather than take them on faith.

The short version:

* The weight **values** you pass are always **inverse variances**
  (`wᵢ = 1/σᵢ²`), never standard deviations.
* The weight **type** decides whether the overall noise scale `σ²` is treated as
  **known** (`vcov = (JᵀWJ)⁻¹`) or **estimated** (`vcov = σ̂² (JᵀWJ)⁻¹`).
* `vcov(fit)` is a parameter **covariance** matrix; `stderror(fit)` is its
  diagonal **standard deviations**.

## Standard deviation vs variance vs covariance

These three live in different units and play different roles. Keeping them
straight removes essentially all the ambiguity.

| Quantity | Symbol | Units | Where it appears in LsqFit |
|:---------|:-------|:------|:---------------------------|
| Standard deviation of observation `i` | `σᵢ` | same as `yᵢ` | *not* passed directly |
| Variance of observation `i` | `σᵢ²` | `yᵢ²` | a **vector** weight is `1/σᵢ²` |
| Covariance of observations | `Σ` | `yᵢ yⱼ` | a **matrix** weight is `Σ⁻¹` |
| Covariance of parameters | `Cov(γ̂)` | `pᵢ pⱼ` | `vcov(fit)` |
| Standard error of parameter `k` | `√Cov(γ̂)ₖₖ` | same as `pₖ` | `stderror(fit)` |

So the data uncertainties go **in** as inverse variances, and the parameter
uncertainties come **out** as a covariance matrix whose square-rooted diagonal
are the standard errors.

!!! warning "Inverse variance, not inverse standard deviation"
    If observation `i` has standard deviation `σᵢ`, its weight is `1/σᵢ²`. A
    common mistake is to pass `1/σᵢ`. The reason `1/σᵢ²` is correct: the weight
    multiplies the **squared** residual, `∑ wᵢ (model−y)ᵢ²`, and the optimal
    (minimum-variance) weighting of a squared residual is the inverse of its
    variance.

## The math

Near the optimum the weighted problem linearises to

```math
\hat{\boldsymbol{h}} = \hat{\boldsymbol{γ}} - \boldsymbol{γ}^*
    \approx -[J'WJ]^{-1} J' W \boldsymbol{r},
```

where `J` is the Jacobian of the model and `W` the weight matrix. With errors
`ε ∼ N(0, Σ)` and the optimal weights `W = Σ⁻¹`, the parameter covariance is

```math
\mathbf{Cov}(\boldsymbol{γ}^*)
    = [J'WJ]^{-1} J'W\,\Sigma\,W'J\,[J'W'J]^{-1}
    = [J'WJ]^{-1}.
```

This is the **known-variance** case: you supplied the exact `Σ⁻¹`, so there is
no scale left to estimate.

If instead you only know the variances **up to a common unknown factor**,
`ε ∼ N(0, σ² W⁻¹)`, then

```math
\mathbf{Cov}(\boldsymbol{γ}^*) = σ^2 [J'WJ]^{-1},
```

and `σ²` is **estimated** from the residuals as `σ̂² = `RSS`/(n-p)` (the mean
squared error, [`LsqFit.mse`](@ref)). Because the same `σ²` multiplies the whole
matrix, this estimator is **scale-invariant**: multiplying every weight by a
constant does not change `vcov`.

Both formulas are correct — they answer different questions. The unweighted fit
is the special case `W = I` of the second formula.

## How `wt` selects the formula

LsqFit picks the formula from the **type** of `wt`. The values are the same
inverse variances in every row below.

| `wt` argument | scale `σ²` | `vcov(fit)` | `nobs` |
|:--------------|:-----------|:------------|:-------|
| *omitted* | estimated | `σ̂² (JᵀJ)⁻¹` | `n` |
| `1 ./ σ.^2` (bare `Vector`) | **known** (`≡1`) | `(JᵀWJ)⁻¹` | `n` |
| `inv(Σ)` (`Matrix`) | **known** (`≡1`) | `(JᵀWJ)⁻¹` | `n` |
| [`AnalyticWeights`](@ref)`(1 ./ σ.^2)` | estimated | `σ̂² (JᵀWJ)⁻¹` | `n` |
| [`FrequencyWeights`](@ref)`(counts)` | estimated | `σ̂² (JᵀWJ)⁻¹` | `∑w` |

The weight types are re-exported from
[StatsBase](https://juliastats.org/StatsBase.jl/stable/weights/) and keep their
StatsBase meaning:

* **`AnalyticWeights`** — *reliability / precision / inverse-variance* weights.
  They describe a **relative** importance, so the common scale is estimated and
  the result is scale-invariant. This matches MATLAB `nlinfit`, Origin and
  LabPlot.
* **`FrequencyWeights`** — integer **counts**: weight `wᵢ` means observation `i`
  was seen `wᵢ` times. Here `nobs = ∑w`, so more counts means more information
  and tighter intervals.
* A **bare vector / matrix** asserts the weights are the *exact* inverse
  (co)variance and the scale is therefore known.

!!! note "Unsupported weight types"
    `ProbabilityWeights` (sampling weights) require a sandwich/survey variance,
    and a generic `Weights` carries no covariance semantics (StatsBase itself
    refuses a bias correction for it). LsqFit therefore throws an informative
    error rather than return a silently wrong covariance.

## Worked example (issue #255)

This is the dataset from the discussion that motivated the typed weights. We fit
`f(x) = A·exp(B·x)` with 5 % relative measurement error.

```@example weights
using LsqFit

model(x, p) = p[1] .* exp.(p[2] .* x)

x = Float64[1, 2, 4, 5, 8]
y = Float64[3, 4, 6, 11, 20]
σ = 0.05 .* y          # known standard deviations
wt = 1 ./ σ.^2         # inverse variances

nothing # hide
```

Treating the weights as **exact** inverse variances (bare vector, scale known):

```@example weights
fit_known = curve_fit(model, x, y, wt, [2.0, 0.3])
coef(fit_known), stderror(fit_known)
```

Treating them as **relative** precisions and estimating the scale
(`AnalyticWeights`):

```@example weights
fit_rel = curve_fit(model, x, y, AnalyticWeights(wt), [2.0, 0.3])
coef(fit_rel), stderror(fit_rel)
```

The point estimates are identical (`A ≈ 2.263`, `B ≈ 0.275`); only the standard
errors differ. The bare-vector errors are about `±0.097` / `±0.009`, while the
`AnalyticWeights` errors are about `±0.258` / `±0.024` — the latter reproduce the
values reported by Origin/LabPlot/mycurvefit for the same data. Neither is a
bug: they answer "what is the uncertainty *given the known noise*?" versus "what
is the uncertainty *if the noise level is itself estimated*?".

### Scale invariance

`AnalyticWeights` describe relative precision, so the overall scale cancels:

```@example weights
se1 = stderror(curve_fit(model, x, y, AnalyticWeights(wt), [2.0, 0.3]))
se2 = stderror(curve_fit(model, x, y, AnalyticWeights(10 .* wt), [2.0, 0.3]))
se1 ≈ se2
```

A bare vector is **not** scale-invariant (multiplying it changes the asserted
variances), and `AnalyticWeights(ones(n))` reproduces the unweighted fit while a
bare vector of ones does not.

### Frequency weights

If each row is really a repeated measurement, use `FrequencyWeights`; `nobs`
becomes the total count:

```@example weights
fit_freq = curve_fit(model, x, y, FrequencyWeights([3, 1, 1, 2, 1]), [2.0, 0.3])
nobs(fit_freq), dof(fit_freq)
```

## Monte-Carlo coverage check

The honest test of an uncertainty estimate is its **coverage**: across many
simulated datasets, a 95 % confidence interval should contain the true parameter
about 95 % of the time. We simulate from a known model with known heteroscedastic
noise and use [`confint`](@ref) (a Student-t interval) for each fit.

```@example weights
using Random

truth = [2.0, 0.25]
xg = collect(range(0.5, 6; length = 15))
rng = MersenneTwister(20240617)
N = 4000

covers(lohi, t) = [lo <= t <= hi for ((lo, hi), t) in zip(lohi, (t[1], t[2]))]

hitK = zeros(2); hitA = zeros(2); hitU = zeros(2)
for _ in 1:N
    sd = 0.10 .* model(xg, truth)               # known standard deviations
    yg = model(xg, truth) .+ sd .* randn(rng, length(xg))
    w  = 1 ./ sd.^2
    fK = curve_fit(model, xg, yg, w, copy(truth))                   # known variance
    fA = curve_fit(model, xg, yg, AnalyticWeights(w), copy(truth))  # estimate scale
    fU = curve_fit(model, xg, yg, copy(truth))                      # unweighted (wrong)
    global hitK += covers(confint(fK; level = 0.95), truth)
    global hitA += covers(confint(fA; level = 0.95), truth)
    global hitU += covers(confint(fU; level = 0.95), truth)
end
(known = round.(100 .* hitK ./ N; digits = 1),
 analytic = round.(100 .* hitA ./ N; digits = 1),
 unweighted = round.(100 .* hitU ./ N; digits = 1))
```

At larger `N` (30 000 in our reference run) the coverages settle to roughly:

| 95 % CIs | A | B |
|:---------|:--|:--|
| bare vector (known variance) | ≈ 97 % | ≈ 97 % |
| `AnalyticWeights` (estimate scale) | ≈ 95 % | ≈ 95 % |
| unweighted (ignores heteroscedasticity) | mis-calibrated | mis-calibrated |

Reading the table:

* **`AnalyticWeights` hit the nominal 95 % almost exactly.** The scale is
  estimated, so `(γ̂ₖ − γₖ)/seₖ` follows Student-t with `dof = n − p`;
  [`confint`](@ref) uses Student-t and the interval is calibrated.
* **The bare vector slightly over-covers — and the cause is the interval
  multiplier, not the covariance.** When the variance is genuinely known there is
  no estimated scale, so the asymptotically-correct multiplier is the normal
  `z = 1.96`. For backwards compatibility [`confint`](@ref) keeps the Student-t
  reference (`t(n−p) = 2.16` here) for a bare vector, inflating the interval by
  `t/z ≈ 1.10` and lifting coverage to ≈ 97 %. This is mildly **conservative**
  (never anti-conservative). The standard errors themselves are exact: swapping
  in the normal quantile recovers 95 % precisely.
* **Ignoring the known heteroscedasticity is mis-calibrated.** This is the case
  that is actually "wrong" — not the choice between the two weighted formulas.

!!! note "Which quantile LsqFit uses"
    `margin_error`/`confint` use Student-t (`dof = n − p`) for the unweighted case
    and for untyped weight inputs (a bare vector or an inverse-covariance matrix),
    preserving historical behaviour. Typed `AbstractWeights` select the
    asymptotically-correct reference instead: Student-t when the scale is
    estimated (`AnalyticWeights`, `FrequencyWeights`) and the standard normal when
    it is known.

This is the empirical version of the conclusion reached in the issue discussion:
*both* weighted interpretations are valid; what matters is matching the
interpretation to what you actually know about the noise.

## Choosing a weight type

* You know each `σᵢ` (e.g. instrument error bars) and trust them as absolute →
  **bare vector `1 ./ σ.^2`** (or `inv(Σ)` for correlated errors).
* You know only the *relative* precision of points, or you want results
  consistent with MATLAB/Origin/LabPlot → **`AnalyticWeights(1 ./ σ.^2)`**.
* Each row is a repeated/aggregated count → **`FrequencyWeights(counts)`**.
* You have nothing better → leave `wt` out (unweighted, scale estimated).
