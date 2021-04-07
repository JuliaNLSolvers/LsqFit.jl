@recipe function f(model::Function, fit::LsqFitResult; significance=0.05, purpose=:neither)
    @series begin
        seriestype --> :line
        label --> "Fit"
        x->model(x, fit.param)
    end
    if purpose in (:confidence, :both)
        @series begin
            seriestype := :line
            seriescolor := :black
            linestyle := :dash
            label := ["Confidence interval" nothing]

            [
             x -> model(x, fit.param) - margin_error(model, x, fit, significance),
             x -> model(x, fit.param) + margin_error(model, x, fit, significance)
            ]
        end
    end
    if purpose in (:prediction, :both)
        @series begin
            seriestype := :line
            seriescolor := :black
            linestyle := :dot
            label := ["Prediction interval" nothing]

            [
             x -> model(x, fit.param) - margin_error(model, x, fit, significance; purpose=:prediction)
             x -> model(x, fit.param) + margin_error(model, x, fit, significance; purpose=:prediction)
            ]
        end
    end
end


"""
```julia
_band(direction, model, x, p_conf)
```

Evaluates `model(x, p)` for each combination `p` in `pconf` (e.g. `[low, low, low]`, `[low, low, high]`...),
and returns the `direction`st return value.

# Example
```julia
_band(minimum, model, x, [(0.0, 1.0), (2.0, 3.0)]) == minimum([model(x, [0.0, 2.0]), model(x, [0.0, 3.0]), model(x, [1.0, 2.0]), model(x, [1.0, 3.0])])
```
"""
function _band(direction, model, x, p_conf)
    l = length(p_conf)
    return direction(begin
                         i = digits(k; base=2, pad=l) .+ 1
                         p = getindex.(p_conf, i)
                         y = model(x, p)
                     end
                     for k in 0:2^l-1
                    )
end

"""
```julia
margin_error(model, x, fit, significance; purpose)
```
Find the width at `x` of the confidence or prediction interval.
"""
function margin_error(model::Function, x, fit::LsqFitResult, alpha=0.05; purpose=:confidence)
    g = p -> ForwardDiff.gradient(p -> model(x, p), fit.param)
    c = g(fit.param)' * estimate_covar(fit) * g(fit.param)
    if purpose === :prediction
        c = c + 1
    end
    dist = TDist(dof(fit))
    critical_values = quantile(dist, 1 - alpha/2)
    return sqrt(c*rss(fit)/dof(fit))*critical_values
end
