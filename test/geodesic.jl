let
    # fitting noisy data to an exponential model
    model(x, p) = @. p[1] * exp(-x * p[2])

    # some example data
    Random.seed!(12345)
    xdata = range(0, stop=10, length=50000)
    ydata = model(xdata, [1.0, 2.0]) + 0.01*randn(length(xdata))
    p0 = [20., 20.]

    # if your model is differentiable, it can be faster and/or more accurate
    # to supply your own jacobian instead of using the finite difference
    function jacobian_model(x,p)
        J = Array{Float64}(undef, length(x), length(p))
        @. J[:,1] = exp(-x*p[2])     #dmodel/dp[1]
        @. @views J[:,2] = -x*p[1]*J[:,1] 
        J
    end

    # a couple notes on the h function
    # - "h" refers to the function, hessians to the array of hessians
    # - we "embed/bake" xdate into function, since we care about the derivatives in "p", always evaluated at the xdata points
    function h!(p,hessians) 
        for i=1:length(xdata)
            hessians[1,1,i] = 0 #Do NOT allocate H with a whole matrix: H = [a b; b c], this would take a lot of memory
            hessians[1,2,i] = (-xdata[i] * exp(-xdata[i] * p[2]))
            hessians[2,1,i] = (-xdata[i] * exp(-xdata[i] * p[2]))
            hessians[2,2,i] = (xdata[i]^2 * p[1] * exp(-xdata[i] * p[2]))
        end
    end


    curve_fit(model, jacobian_model, xdata, ydata, p0; maxIter=1); #warmup
    curve_fit(model, jacobian_model, h!, xdata, ydata, p0; maxIter=1);

    println("--------------\nPerformance of curve_fit vs geo")

    println("\t Non-inplace")
    fit = @time curve_fit(model, jacobian_model, xdata, ydata, p0; maxIter=100)
    @test fit.converged


    println("\t Geodesic")
    fit_geo = @time curve_fit(model, jacobian_model, h!, xdata, ydata, p0; maxIter=100)
    @test fit_geo.converged

    @test maximum(fit.param-fit_geo.param) < 1e-8


end

