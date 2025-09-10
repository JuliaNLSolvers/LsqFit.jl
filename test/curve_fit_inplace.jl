using LsqFit, Test, StableRNGs, LinearAlgebra
@testset "inplace" begin
    # fitting noisy data to an exponential model
    # TODO: Change to `.-x` when 0.5 support is dropped
    @. model(x, p) = p[1] * exp(-x * p[2])
    model_inplace(F, x, p) = (@. F = p[1] * exp(-x * p[2]))

    # some example data
    rng = StableRNG(123)
    xdata = range(0, stop = 10, length = 500000)
    ydata = model(xdata, [1.0, 2.0]) + 0.01 * randn(rng, length(xdata))
    p0 = [0.5, 0.5]

    # if your model is differentiable, it can be faster and/or more accurate
    # to supply your own jacobian instead of using the finite difference
    function jacobian_model(x, p)
        J = Array{Float64}(undef, length(x), length(p))
        @. J[:, 1] = exp(-x * p[2])     #dmodel/dp[1]
        @. @views J[:, 2] = -x * p[1] * J[:, 1]
        J
    end

    function jacobian_model_inplace(J::Array{Float64,2}, x, p)
        @. J[:, 1] = exp(-x * p[2])     #dmodel/dp[1]
        @. @views J[:, 2] = -x * p[1] * J[:, 1]
    end


    f(p) = model(xdata, p) - ydata
    g(p) = jacobian_model(xdata, p)
    df = OnceDifferentiable(f, g, p0, similar(ydata); inplace = false)
    evalf(x) = NLSolversBase.value!!(df, x)
    evalg(x) = NLSolversBase.jacobian!!(df, x)
    r = evalf(p0)
    j = evalg(p0)

    f_inplace = (F, p) -> (model_inplace(F, xdata, p); @. F = F - ydata)
    g_inplace = (G, p) -> jacobian_model_inplace(G, xdata, p)
    df_inplace =
        OnceDifferentiable(f_inplace, g_inplace, p0, similar(ydata); inplace = true)
    evalf_inplace(x) = NLSolversBase.value!!(df_inplace, x)
    evalg_inplace(x) = NLSolversBase.jacobian!!(df_inplace, x)
    r_inplace = evalf_inplace(p0)
    j_inplace = evalg_inplace(p0)


    @test r == r_inplace
    @test j == j_inplace

    println("--------------\nPerformance of non-inplace")
    println("\t Evaluation function")

    stop = 8 #8 because the tests afterwards will call the eval function 8 or 9 times, so it makes it easy to compare
    step = 1

    @time for i in range(0, stop = stop, step = step)
        evalf(p0)
    end

    println("\t Jacobian function")
    @time for i in range(0, stop = stop, step = step)
        evalg(p0)
    end

    println("--------------\nPerformance of inplace")
    println("\t Evaluation function")
    @time for i in range(0, stop = stop, step = step)
        evalf_inplace(p0)
    end

    println("\t Jacobian function")
    @time for i in range(0, stop = stop, step = step)
        evalg_inplace(p0)
    end


    curve_fit(model, xdata, ydata, p0; maxIter = 100) #warmup
    curve_fit(model_inplace, xdata, ydata, p0; inplace = true, maxIter = 100)

    #explicit jac
    curve_fit(model, jacobian_model, xdata, ydata, p0; maxIter = 100)
    curve_fit(
        model_inplace,
        jacobian_model_inplace,
        xdata,
        ydata,
        p0;
        inplace = true,
        maxIter = 100,
    )



    println("--------------\nPerformance of curve_fit")

    println("\t Non-inplace")
    fit = @time curve_fit(model, xdata, ydata, p0; maxIter = 100)
    @test fit.converged

    println("\t Inplace")
    fit_inplace =
        @time curve_fit(model_inplace, xdata, ydata, p0; inplace = true, maxIter = 100)
    @test fit_inplace.converged

    @test fit_inplace.param == fit.param


    println("\t Non-inplace with jacobian")
    fit_jac = @time curve_fit(model, jacobian_model, xdata, ydata, p0; maxIter = 100)
    @test fit_jac.converged

    println("\t Inplace with jacobian")
    fit_inplace_jac = @time curve_fit(
        model_inplace,
        jacobian_model_inplace,
        xdata,
        ydata,
        p0;
        inplace = true,
        maxIter = 100,
    )
    @test fit_inplace_jac.converged

    @test fit_jac.param == fit_inplace_jac.param




    # some example data
    yvars = 1e-6 * rand(rng, length(xdata))
    ydata = model(xdata, [1.0, 2.0]) + sqrt.(yvars) .* randn(rng, length(xdata))

    println("--------------\nPerformance of curve_fit with weights")

    curve_fit(model, xdata, ydata, 1 ./ yvars, [0.5, 0.5])
    curve_fit(
        model_inplace,
        xdata,
        ydata,
        1 ./ yvars,
        [0.5, 0.5];
        inplace = true,
        maxIter = 100,
    )

    curve_fit(model, jacobian_model, xdata, ydata, 1 ./ yvars, [0.5, 0.5])
    curve_fit(
        model_inplace,
        jacobian_model_inplace,
        xdata,
        ydata,
        1 ./ yvars,
        [0.5, 0.5];
        inplace = true,
        maxIter = 100,
    )


    println("\t Non-inplace with weights")
    fit_wt = @time curve_fit(
        model,
        jacobian_model,
        xdata,
        ydata,
        1 ./ yvars,
        [0.5, 0.5];
        maxIter = 100,
    )
    @test fit_wt.converged

    @testset "bad input" begin
        xxdata = copy(collect(xdata))
        yydata = copy(ydata)
        WWT = 1 ./ sqrt.(yvars)
        x1 = xxdata[1]
        y1 = yydata[1]
        wt1 = WWT[1]
        for x in (x1, Inf, -Inf, NaN), y in (y1, Inf, -Inf, NaN), wt in (wt1, Inf, -Inf, NaN)

            xxdata[1] = x
            yydata[1] = y
            WWT[1] = wt
            if x == x1 && y == y1 && wt == wt1
                @test_nowarn curve_fit(model, jacobian_model, xxdata, yydata, WWT, [0.5, 0.5]; maxIter=100)
            else
                @test_throws ErrorException curve_fit(model, jacobian_model, xxdata, yydata, WWT, [0.5, 0.5]; maxIter=100)
            end
        end
    end

    println("\t Inplace with weights")
    fit_inplace_wt = @time curve_fit(
        model_inplace,
        xdata,
        ydata,
        1 ./ yvars,
        [0.5, 0.5];
        inplace = true,
        maxIter = 100,
    )
    @test fit_inplace_wt.converged

    @test maximum(abs.(fit_wt.param - fit_inplace_wt.param)) < 1e-15


    println("\t Non-inplace with jacobian with weights")
    fit_wt_jac = @time curve_fit(
        model,
        jacobian_model,
        xdata,
        ydata,
        1 ./ yvars,
        [0.5, 0.5];
        maxIter = 100,
    )
    @test fit_wt_jac.converged

    println("\t Inplace with jacobian with weights")
    fit_inplace_wt_jac = @time curve_fit(
        model_inplace,
        jacobian_model_inplace,
        xdata,
        ydata,
        1 ./ yvars,
        [0.5, 0.5];
        inplace = true,
        maxIter = 100,
    )
    @test fit_inplace_wt_jac.converged

    @test maximum(abs.(fit_wt_jac.param - fit_inplace_wt_jac.param)) < 1e-15

end
