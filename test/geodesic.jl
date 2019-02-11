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

    # a couple notes on the Avv function:
    # - the basic idea is to see the model output as simply a collection of functions: f1...fm
    # - then Avv return an array of size m, where each elements corresponds to
    # v'H(p)v, with H an n*n Hessian matrix of the m-th function, with n the size of p
    function Avv!(p,v,dir_deriv)
        for i=1:length(xdata)
            #compute all the elements of the H matrix
            h11 = 0
            h12 = (-xdata[i] * exp(-xdata[i] * p[2]))
            #h21 = h12
            h22 = (xdata[i]^2 * p[1] * exp(-xdata[i] * p[2]))
            v1 = v[1]
            v2 = v[2]
            # manually compute v'Hv. This whole process might seem cumbersome, but 
            # allocating temporary matrices quickly becomes REALLY expensive and might even 
            # render the use of geodesic acceleration terribly inefficient  
            dir_deriv[i] = h11*v1^2 + 2*h12*v1*v2 + h22*v2^2

        end
    end 


    curve_fit(model, jacobian_model, xdata, ydata, p0; maxIter=1); #warmup
    curve_fit(model, jacobian_model, xdata, ydata, p0; maxIter=1, avv! = Avv!);

    println("--------------\nPerformance of curve_fit vs geo")

    println("\t Non-inplace")
    fit = @time curve_fit(model, jacobian_model, xdata, ydata, p0; maxIter=100)
    @test fit.converged


    println("\t Geodesic")
    fit_geo = @time curve_fit(model, jacobian_model, xdata, ydata, p0; maxIter=100, avv! = Avv!)
    @test fit_geo.converged

    @test maximum(fit.param-fit_geo.param) < 1e-8


    #with noise
    yvars = 1e-6*rand(length(xdata))
    ydata = model(xdata, [1.0, 2.0]) + sqrt.(yvars) .* randn(length(xdata))

    #warm up
    curve_fit(model, jacobian_model, xdata, ydata, 1 ./ yvars, p0; maxIter=1)
    curve_fit(model, jacobian_model, xdata, ydata,  1 ./ yvars, p0; maxIter=1, avv! = Avv!)

    println("--------------\nPerformance of curve_fit vs geo with weights")

    println("\t Non-inplace")
    fit_wt = @time curve_fit(model, jacobian_model, xdata, ydata, 1 ./ yvars, p0; maxIter=100)
    @test fit_wt.converged


    println("\t Geodesic")
    fit_geo_wt = @time curve_fit(model, jacobian_model, xdata, ydata,  1 ./ yvars, p0; maxIter=100, avv! = Avv!)
    @test fit_geo_wt.converged

    @test maximum(fit_wt.param-fit_geo_wt.param) < 1e-8


end

