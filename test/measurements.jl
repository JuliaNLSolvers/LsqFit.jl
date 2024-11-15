using LsqFit
using Measurements
using StableRNGs

@testset "MeasurementsExt" begin
    model(x, p) = p[1] .* exp.(-x .* p[2])
    ptrue = [10, 0.3]
    x = LinRange(0, 2, 50)
    y0 = model(x, ptrue)
    σ = rand(1:5, 50)
    y = y0 .± σ
    wt = σ .^ -2
    fit0 = curve_fit(model, x, y0, wt, ptrue) # fit to data using weights
    fit1 = curve_fit(model, x, y, ptrue) # fit to data using Measurements
    @test coef(fit0) ≈ coef(fit1)

    # some example data
    rng = StableRNG(125)
    x = range(0, stop = 10, length = 20)
    y0 = model(x, [1.0, 2.0]) + 0.01 * randn(rng, length(x))
    σ = 0.01
    y = y0 .± σ
    wt = ones(length(x)) * σ .^ -2
    p0 = [0.5, 0.5]

    for ad in (:finite, :forward, :forwarddiff)
        fit0 = curve_fit(model, x, y0, wt, p0; autodiff = ad)
        fit1 = curve_fit(model, x, y, p0; autodiff = ad)
        for fit in (fit0, fit1)
            @test norm(fit.param - [1.0, 2.0]) < 0.05
            @test fit.converged

            # can also get error estimates on the fit parameters
            errors = margin_error(fit, 0.05)
            @test norm(errors - [0.021, 0.093]) < 0.01
        end
    end
    # if your model is differentiable, it can be faster and/or more accurate
    # to supply your own jacobian instead of using the finite difference
    function jacobian_model(x, p)
        J = Array{Float64}(undef, length(x), length(p))
        J[:, 1] = exp.(-x .* p[2])             #dmodel/dp[1]
        J[:, 2] = -x .* p[1] .* J[:, 1]           #dmodel/dp[2]
        J
    end
    jacobian_fit0 = curve_fit(model, jacobian_model, x, y0, wt, p0;show_trace=true)
    jacobian_fit1 = curve_fit(model, jacobian_model, x, y, p0;show_trace=true)
    for jacobian_fit in (jacobian_fit0, jacobian_fit1)
        @test norm(jacobian_fit.param - [1.0, 2.0]) < 0.05
        @test jacobian_fit.converged
    end
end