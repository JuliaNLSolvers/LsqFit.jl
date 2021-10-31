@testset "curve_fit" begin
    # before testing the model, check whether missing/null data is rejected
    tdata = [rand(1:10, 5)..., missing]
    @test_throws ErrorException("x data contains `missing`, `Inf` or `NaN` values and a fit cannot be performed") LsqFit.check_data_health(tdata, tdata)
    tdata = [rand(1:10, 5)..., Inf]
    @test_throws ErrorException("x data contains `missing`, `Inf` or `NaN` values and a fit cannot be performed") LsqFit.check_data_health(tdata, tdata)
    tdata = [rand(1:10, 5)..., NaN]
    @test_throws ErrorException("x data contains `missing`, `Inf` or `NaN` values and a fit cannot be performed") LsqFit.check_data_health(tdata, tdata)

    # fitting noisy data to an exponential model
    model(x, p) = p[1] * exp(-x * p[2])

    # some example data
    Random.seed!(12345)
    xdata = range(0, stop=10, length=20)
    ydata = model.(xdata, Ref([1.0, 2.0])) + 0.01*randn(length(xdata))
    p0 = [0.5, 0.5]

    for ad in (:finite, :forward, :forwarddiff)
        fit = curve_fit(model, xdata, ydata, p0; autodiff = ad)
        @test norm(fit.param - [1.0, 2.0]) < 0.05
        @test fit.converged

        # can also get error estimates on the fit parameters
        errors = margin_error(fit, 0.1)
        @test norm(errors - [0.017, 0.075]) < 0.015
    end
    # if your model is differentiable, it can be faster and/or more accurate
    # to supply your own jacobian instead of using the finite difference
    function jacobian_model(x,p)
        J = Array{Float64}(undef, length(p))
        J[1] = exp(-x*p[2])             #dmodel/dp[1]
        J[2] = -x*p[1]*J[1]           #dmodel/dp[2]
        J
    end
    jacobian_fit = curve_fit(model, jacobian_model, xdata, ydata, p0)
    @test norm(jacobian_fit.param - [1.0, 2.0]) < 0.05
    @test jacobian_fit.converged

    # some example data
    yvars = 1e-6*rand(length(xdata))
    ydata = model.(xdata, Ref([1.0, 2.0])) + sqrt.(yvars) .* randn(length(xdata))

    fit = curve_fit(model, xdata, ydata, 1 ./ yvars, [0.5, 0.5])
    println("norm(fit.param - [1.0, 2.0]) < 0.05 ? ", norm(fit.param - [1.0, 2.0]))
    @test norm(fit.param - [1.0, 2.0]) < 0.05
    @test fit.converged

    # test matrix valued weights ( #161 )
    weights = LinearAlgebra.diagm(1 ./ yvars)
    fit_matrixweights = curve_fit(model, xdata, ydata, weights, [0.5, 0.5])
    @test fit.param == fit_matrixweights.param

    # can also get error estimates on the fit parameters
    errors = margin_error(fit, 0.1)
    println("norm(errors - [0.017, 0.075]) < 0.1 ?", norm(errors - [0.017, 0.075]))
    @test norm(errors - [0.017, 0.075]) < 0.1

    # test with user-supplied jacobian and weights
    fit = curve_fit(model, jacobian_model, xdata, ydata, 1 ./ yvars, p0)
    println("norm(fit.param - [1.0, 2.0]) < 0.05 ? ", norm(fit.param - [1.0, 2.0]))
    @test norm(fit.param - [1.0, 2.0]) < 0.05
    @test fit.converged

    # Parameters can also be inferred using arbitrary precision
    fit = curve_fit(model, xdata, ydata, 1 ./ yvars, BigFloat.(p0); x_tol=1e-20, g_tol=1e-20)
    @test fit.converged
    fit = curve_fit(model, jacobian_model, xdata, ydata, 1 ./ yvars, BigFloat.(p0); x_tol=1e-20, g_tol=1e-20)
    @test fit.converged

    curve_fit(model, jacobian_model, xdata, ydata, 1 ./ yvars, [0.5, 0.5]; tau=0.0001)
end
