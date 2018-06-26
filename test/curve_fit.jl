let
    # fitting noisy data to an exponential model
    # TODO: Change to `.-x` when 0.5 support is dropped
    model(x, p) = p[1] .* exp.(-x .* p[2])

    # some example data
    srand(12345)
    xdata = linspace(0,10,20)
    ydata = model(xdata, [1.0, 2.0]) + 0.01*randn(length(xdata))
    p0 = [0.5, 0.5]

    fit = curve_fit(model, xdata, ydata, p0)
    @assert norm(fit.param - [1.0, 2.0]) < 0.05
    @test fit.converged

    # can also get error estimates on the fit parameters
    errors = margin_error(fit, 0.1)
    @assert norm(errors - [0.017, 0.075]) < 0.01

    # if your model is differentiable, it can be faster and/or more accurate
    # to supply your own jacobian instead of using the finite difference
    function jacobian_model(x,p)
        J = Array{Float64}(length(x),length(p))
        J[:,1] = exp.(-x.*p[2])     #dmodel/dp[1]
        J[:,2] = -x.*p[1].*J[:,1]           #dmodel/dp[2]
        J
    end
    jacobian_fit = curve_fit(model, jacobian_model, xdata, ydata, p0)
    @assert norm(jacobian_fit.param - [1.0, 2.0]) < 0.05
    @test jacobian_fit.converged

    # some example data
    ystdd = 1e-6*rand(length(xdata))
    ydata = model(xdata, [1.0, 2.0]) + ystdd .* randn(length(xdata))

    fit = curve_fit(model, xdata, ydata, ystdd, [0.5, 0.5])
    println("norm(fit.param - [1.0, 2.0]) < 0.05 ? ", norm(fit.param - [1.0, 2.0]))
    @assert norm(fit.param - [1.0, 2.0]) < 0.05
    @test fit.converged

    # can also get error estimates on the fit parameters
    errors = margin_error(fit, 0.1)
    println("norm(errors - [0.017, 0.075]) < 0.1 ?", norm(errors - [0.017, 0.075]))
    @assert norm(errors - [0.017, 0.075]) < 0.1

    # test with user-supplied jacobian and covariance matrix
    fit = curve_fit(model, jacobian_model, xdata, ydata, diagm(ystdd.^2), [0.5, 0.5])
    println("norm(fit.param - [1.0, 2.0]) < 0.05 ? ", norm(fit.param - [1.0, 2.0]))
    @assert norm(fit.param - [1.0, 2.0]) < 0.05
    @test fit.converged
end
