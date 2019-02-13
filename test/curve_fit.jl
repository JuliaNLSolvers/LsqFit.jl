
@testset "#110" begin
    model(x,p) = @. p[1]*exp(-x/p[2])+p[3]
    function jacobian_model(x,p)
      J = Array{Float64}(undef, length(x),length(p))
      ip2 = 1/p[2]
      J[:,2] = x.*ip2
      J[:,1] = exp.(-J[:,2])        #d/dp1 = exp(-x./p2)
      J[:,2] .*= p[1].*ip2.*J[:,1]  #d/dp2 = x*p1/p2^2*exp(-x/p2)
      J[:,3] .= 1.0                 #d/dp3 = ones(length(x))
      J
    end

    x = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 388.0, 389.0, 390.0, 391.0, 392.0, 393.0, 394.0, 395.0, 396.0, 397.0, 398.0, 399.0, 400.0]

    y = [8356.0, 8167.0, 7843.0, 8012.0, 7834.0, 7793.0, 7939.0, 8101.0, 7773.0, 7434.0, 7470.0, 7666.0, 7460.0, 7406.0, 7449.0, 7284.0, 7353.0, 7344.0, 7458.0, 7097.0, 7205.0, 7344.0, 7207.0, 7121.0, 7168.0, 7259.0, 7340.0, 6948.0, 6885.0, 7159.0, 7247.0, 7182.0, 7105.0, 6981.0, 6897.0, 7081.0, 6884.0, 6963.0, 7019.0, 6954.0, 6947.0, 6427.0, 6225.0, 6248.0, 6331.0, 6196.0, 6312.0, 6158.0, 6289.0, 6343.0, 6350.0, 6154.0, 6479.0, 6164.0]

    w = [0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0243902, 0.0769231, 0.0769231, 0.0769231, 0.0769231, 0.0769231, 0.0769231, 0.0769231, 0.0769231, 0.0769231, 0.0769231, 0.0769231, 0.0769231, 0.0769231]

    p0 = [4.0, 396.0, 400.0]

    fit = curve_fit(model, jacobian_model, x, y, w, p0)
    p_true =  [7.518878254077602e6, 6.330943450510559e9, -7.512052070395852e6]
    @test p_true ≈ coef(fit)
end

let
    # fitting noisy data to an exponential model
    # TODO: Change to `.-x` when 0.5 support is dropped
    model(x, p) = p[1] .* exp.(-x .* p[2])

    # some example data
    Random.seed!(12345)
    xdata = range(0, stop=10, length=20)
    ydata = model(xdata, [1.0, 2.0]) + 0.01*randn(length(xdata))
    p0 = [0.5, 0.5]

    for ad in (:finite, :forward, :forwarddiff)
        fit = curve_fit(model, xdata, ydata, p0; autodiff = ad)
        @assert norm(fit.param - [1.0, 2.0]) < 0.05
        @test fit.converged

        # can also get error estimates on the fit parameters
        errors = margin_error(fit, 0.1)
        @assert norm(errors - [0.017, 0.075]) < 0.01
    end
    # if your model is differentiable, it can be faster and/or more accurate
    # to supply your own jacobian instead of using the finite difference
    function jacobian_model(x,p)
        J = Array{Float64}(undef, length(x), length(p))
        J[:,1] = exp.(-x.*p[2])     #dmodel/dp[1]
        J[:,2] = -x.*p[1].*J[:,1]           #dmodel/dp[2]
        J
    end
    jacobian_fit = curve_fit(model, jacobian_model, xdata, ydata, p0)
    @assert norm(jacobian_fit.param - [1.0, 2.0]) < 0.05
    @test jacobian_fit.converged

    # some example data
    yvars = 1e-6*rand(length(xdata))
    ydata = model(xdata, [1.0, 2.0]) + sqrt.(yvars) .* randn(length(xdata))

    fit = curve_fit(model, xdata, ydata, 1 ./ yvars, [0.5, 0.5])
    println("norm(fit.param - [1.0, 2.0]) < 0.05 ? ", norm(fit.param - [1.0, 2.0]))
    @assert norm(fit.param - [1.0, 2.0]) < 0.05
    @test fit.converged

    # can also get error estimates on the fit parameters
    errors = margin_error(fit, 0.1)
    println("norm(errors - [0.017, 0.075]) < 0.1 ?", norm(errors - [0.017, 0.075]))
    @assert norm(errors - [0.017, 0.075]) < 0.1

    # test with user-supplied jacobian and weights
    fit = curve_fit(model, jacobian_model, xdata, ydata, 1 ./ yvars, [0.5, 0.5])
    println("norm(fit.param - [1.0, 2.0]) < 0.05 ? ", norm(fit.param - [1.0, 2.0]))
    @assert norm(fit.param - [1.0, 2.0]) < 0.05
    @test fit.converged
end
