"""
    Pointwise(f)

Wrap a model written for a single observation so that [`curve_fit`](@ref) evaluates
it once per observation. `f(xᵢ, p)` takes one observation `xᵢ` and the parameter
vector `p`, and returns a number.

An observation is an element of a vector `xdata`, or a row of a matrix `xdata` (the
layout of the multivariate example in the README). A model of several independent
variables can therefore take a matrix with one row per observation, or a vector of
tuples. The number of observations must equal `length(ydata)`.

`Pointwise(f)(xdata, p)` returns the vector of predictions, so the wrapped model also
predicts at new points. The wrapper fills the residual in place when `curve_fit` is
called with `inplace = true`.

# Example
```julia
m(x, p) = p[1] * exp(-p[2] * x)
fit = curve_fit(Pointwise(m), xdata, ydata, p0)
Pointwise(m)(xnew, coef(fit)) # predictions at new points
```

See also [`PointwiseJacobian`](@ref).
"""
struct Pointwise{F}
    f::F
end

"""
    PointwiseJacobian(g)

Wrap the gradient of a [`Pointwise`](@ref) model so that it can be passed to
[`curve_fit`](@ref) as the Jacobian. `g(xᵢ, p)` returns the gradient of the model
with respect to `p` at one observation: a number for a one-parameter model, or a
tuple or vector of length `length(p)`. The gradients are the rows of the Jacobian.

A tuple gradient avoids allocating a vector for each observation.

# Example
```julia
m(x, p) = p[1] * exp(-p[2] * x)
g(x, p) = (exp(-p[2] * x), -x * p[1] * exp(-p[2] * x))
fit = curve_fit(Pointwise(m), PointwiseJacobian(g), xdata, ydata, p0)
```
"""
struct PointwiseJacobian{G}
    g::G
end

function (m::Pointwise)(xdata::AbstractArray, p)
    return map(xᵢ -> _pointwise_value(m.f(xᵢ, p)), _observations(xdata))
end

function (m::Pointwise)(F::AbstractVector, xdata::AbstractArray, p)
    obs = _observations(xdata)
    _check_nobs(length(F), obs)
    for (i, xᵢ) in zip(eachindex(F), obs)
        F[i] = _pointwise_value(m.f(xᵢ, p))
    end
    return F
end

function (jac::PointwiseJacobian)(xdata::AbstractArray, p)
    J = similar(p, length(_observations(xdata)), length(p))
    return jac(J, xdata, p)
end

function (jac::PointwiseJacobian)(J::AbstractMatrix, xdata::AbstractArray, p)
    obs = _observations(xdata)
    _check_nobs(size(J, 1), obs)
    for (i, xᵢ) in zip(axes(J, 1), obs)
        gᵢ = jac.g(xᵢ, p)
        if length(gᵢ) != size(J, 2)
            throw(
                DimensionMismatch(
                    "The gradient of a `PointwiseJacobian` must have one entry per parameter: got $(length(gᵢ)) entries for $(size(J, 2)) parameters",
                ),
            )
        end
        for (j, gᵢⱼ) in zip(axes(J, 2), gᵢ)
            J[i, j] = gᵢⱼ
        end
    end
    return J
end

_observations(xdata::AbstractVector) = xdata
_observations(xdata::AbstractMatrix) = eachrow(xdata)
function _observations(xdata::AbstractArray)
    throw(
        ArgumentError(
            "`xdata` for a `Pointwise` model must be a vector of observations or a matrix with one observation per row, got a $(ndims(xdata))-dimensional array",
        ),
    )
end

function _check_nobs(n, obs)
    if length(obs) != n
        throw(
            DimensionMismatch(
                "`xdata` has $(length(obs)) observations but `ydata` has $n values (a matrix `xdata` has one observation per row)",
            ),
        )
    end
    return nothing
end

_pointwise_value(y::Number) = y
function _pointwise_value(y)
    throw(
        ArgumentError(
            "A `Pointwise` model must return a number for each observation, got a `$(typeof(y))`. Wrap a one-element result in `only`.",
        ),
    )
end
