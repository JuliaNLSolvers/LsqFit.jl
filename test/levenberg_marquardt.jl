@testset "optimization" begin
    function f_lm(x)
      [x[1], 2.0 - x[2]]
    end
    function g_lm(x)
      [1.0 0.0; 0.0 -1.0]
    end

    initial_x = [100.0, 100.0]

    R1 = OnceDifferentiable(f_lm, g_lm, zeros(2), zeros(2); inplace = false)
    results = LsqFit.levenberg_marquardt(R1, initial_x)
    @assert norm(OptimBase.minimizer(results) - [0.0, 2.0]) < 0.01


    function rosenbrock_res(r, x)
        r[1] = 10.0 * (x[2] - x[1]^2 )
        r[2] =  1.0 - x[1]
        return r
    end

    function rosenbrock_jac(j, x)
        j[1, 1] = -20.0 * x[1]
        j[1, 2] =  10.0
        j[2, 1] =  -1.0
        j[2, 2] =   0.0
        return j
    end


    initial_xrb = [-1.2, 1.0]

    R2 = OnceDifferentiable(rosenbrock_res, rosenbrock_jac, zeros(2), zeros(2); inplace = true)

    results = LsqFit.levenberg_marquardt(R2, initial_xrb)

    @assert norm(OptimBase.minimizer(results) - [1.0, 1.0]) < 0.01

    # check estimate is within the bound PR #278
    result = LsqFit.levenberg_marquardt(R2, [150.0, 150.0]; lower = [10.0, 10.0], upper = [200.0, 200.0])
    @test OptimBase.minimizer(result)[1] >= 10.0
    @test OptimBase.minimizer(result)[2] >= 10.0




    # tests for #178, taken from LsqFit.jl, but stripped
    let
        Random.seed!(12345)

        # TODO: Change to `.-x` when 0.5 support is dropped
        model(x, p) = p[1] .* exp.(-x .* p[2])
        _nobs = 20
        xdata = range(0, stop = 10, length = _nobs)
        ydata = model(xdata, [1.0 2.0]) + 0.01*randn(_nobs)

        f_lsq = p ->  model(xdata, p) - ydata
        g_lsq = p -> FiniteDiff.finite_difference_jacobian(f_lsq, x)

        R3 = OnceDifferentiable(f_lsq, g_lsq, zeros(2), zeros(_nobs); inplace = false)
        results = LsqFit.levenberg_marquardt(R3, [0.5, 0.5])

        @assert norm(OptimBase.minimizer(results) - [1.0, 2.0]) < 0.05
    end

    let
        Random.seed!(12345)

        # TODO: Change to `.-x` when 0.5 support is dropped
        model(x, p) = p[1] .* exp.(-x ./ p[2]) .+ p[3]

        xdata = 1:100
        p_generate = [10.0, 10.0, 10.0]
        ydata = model(xdata, p_generate) + 0.1*randn(length(xdata))

        # TODO: Change to `model.(xdata, p) .- ydata` when 0.4 support is dropped
        f_lsq = p -> model(xdata, p) - ydata
        g_lsq = p -> FiniteDiff.finite_difference_jacobian(f_lsq, x)

        R4 = OnceDifferentiable(f_lsq, g_lsq, similar(p_generate), zeros(length(xdata)); inplace = false)

        # tests for box constraints, PR #196
        @test_throws ArgumentError LsqFit.levenberg_marquardt(R4, [15.0, 15.0, 15.0], lower=[5.0, 11.0])
        @test_throws ArgumentError LsqFit.levenberg_marquardt(R4, [5.0, 5.0, 5.0], upper=[15.0, 9.0])
        @test_throws ArgumentError LsqFit.levenberg_marquardt(R4, [15.0, 10.0, 15.0], lower=[5.0, 11.0, 5.0])
        @test_throws ArgumentError LsqFit.levenberg_marquardt(R4, [5.0, 10.0, 5.0], upper=[15.0, 9.0, 15.0])


        lower=[5.0, 11.0, 5.0]
        results = LsqFit.levenberg_marquardt(R4, [15.0, 15.0, 15.0], lower=lower)
        OptimBase.minimizer(results)
        @test OptimBase.converged(results)
        @test all(OptimBase.minimizer(results) .>= lower)

        upper=[15.0, 9.0, 15.0]
        results = LsqFit.levenberg_marquardt(R4, [5.0, 5.0, 5.0], upper=upper)
        OptimBase.minimizer(results)
        @test OptimBase.converged(results)
        @test all(OptimBase.minimizer(results) .<= upper)

        # tests for PR #267
        LsqFit.levenberg_marquardt(R4, [15.0, 15.0, 15.0], show_trace=true)
    end
end
