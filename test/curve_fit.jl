using LsqFit, Test, StableRNGs, LinearAlgebra
@testset "curve fit" begin
    # before testing the model, check whether missing/null data is rejected
    tdata = [rand(1:10, 5)..., missing]
    @test_throws ErrorException(
        "The independent variable (`x`) contains `missing` values and a fit cannot be performed",
    ) LsqFit.check_data_health(tdata, tdata)
    tdata = [rand(1:10, 5)..., Inf]
    @test_throws ErrorException(
        "The independent variable (`x`) contains non-finite (e.g. `Inf`, `NaN`) values and a fit cannot be performed",
    ) LsqFit.check_data_health(tdata, tdata)
    tdata = [rand(1:10, 5)..., NaN]
    @test_throws ErrorException(
        "The independent variable (`x`) contains non-finite (e.g. `Inf`, `NaN`) values and a fit cannot be performed",
    ) LsqFit.check_data_health(tdata, tdata)

    # fitting noisy data to an exponential model
    # TODO: Change to `.-x` when 0.5 support is dropped
    model(x, p) = p[1] .* exp.(-x .* p[2])

    for T in (Float64, BigFloat)
        # fitting noisy data to an exponential model
        # TODO: Change to `.-x` when 0.5 support is dropped
        # model(x, p) = p[1] .* exp.(-x .* p[2])

        # some example data
        rng = StableRNG(125)
        xdata = range(0, stop = 10, length = 20)
        ydata = T.(model(xdata, [1.0, 2.0]) + 0.01 * randn(rng, length(xdata)))
        p0 = T.([0.5, 0.5])

        for ad in (AutoFiniteDiff(fdjtype = Val(:central)), AutoForwardDiff())
            fit = curve_fit(model, xdata, ydata, p0; autodiff = ad)
            @test norm(fit.param - [1.0, 2.0]) < 0.05
            @test fit.converged

            # can also get error estimates on the fit parameters
            errors = margin_error(fit, 0.05)
            @test norm(errors - [0.025, 0.11]) < 0.01
        end
        # if your model is differentiable, it can be faster and/or more accurate
        # to supply your own jacobian instead of using the finite difference
        function jacobian_model(x, p)
            J = Array{Float64}(undef, length(x), length(p))
            J[:, 1] = exp.(-x .* p[2])             #dmodel/dp[1]
            J[:, 2] = -x .* p[1] .* J[:, 1]           #dmodel/dp[2]
            J
        end
        jacobian_fit = curve_fit(model, jacobian_model, xdata, ydata, p0; show_trace = true)
        @test norm(jacobian_fit.param - [1.0, 2.0]) < 0.05
        @test jacobian_fit.converged
        @testset "#195" begin
            @test length(jacobian_fit.trace) > 1
            @test jacobian_fit.trace[end].metadata["dx"][1] !=
                  jacobian_fit.trace[end-1].metadata["dx"][1]
        end
        # some example data
        yvars = T.(1e-6 * rand(rng, length(xdata)))
        ydata = T.(model(xdata, [1.0, 2.0]) + sqrt.(yvars) .* randn(rng, length(xdata)))

        fit = curve_fit(model, xdata, ydata, 1 ./ yvars, T.([0.5, 0.5]))

        @test norm(fit.param - [1.0, 2.0]) < 0.05
        @test fit.converged

        # test matrix valued weights ( #161 )
        weights = LinearAlgebra.diagm(1 ./ yvars)
        fit_matrixweights = curve_fit(model, xdata, ydata, weights, T.([0.5, 0.5]))
        @test fit.param == fit_matrixweights.param

        # can also get error estimates on the fit parameters
        errors = margin_error(fit, T(0.1))

        @test norm(errors - [0.017, 0.075]) < 0.1

        # test with user-supplied jacobian and weights
        fit = curve_fit(model, jacobian_model, xdata, ydata, 1 ./ yvars, p0)
        println("norm(fit.param - [1.0, 2.0]) < 0.05 ? ", norm(fit.param - [1.0, 2.0]))
        @test norm(fit.param - [1.0, 2.0]) < 0.05
        @test fit.converged

        # Parameters can also be inferred using arbitrary precision
        fit = curve_fit(
            model,
            xdata,
            ydata,
            1 ./ yvars,
            BigFloat.(p0);
            x_tol = T(1e-20),
            g_tol = T(1e-20),
        )
        @test fit.converged
        fit = curve_fit(
            model,
            jacobian_model,
            xdata,
            ydata,
            1 ./ yvars,
            BigFloat.(p0);
            x_tol = T(1e-20),
            g_tol = T(1e-20),
        )
        @test fit.converged

        curve_fit(model, jacobian_model, xdata, ydata, 1 ./ yvars, [0.5, 0.5]; tau = 0.0001)
    end

end

@testset "autodiff" begin
    model(x, p) = p[1] .* exp.(-x .* p[2])
    rng = StableRNG(125)
    xdata = range(0, stop = 10, length = 20)
    ydata = model(xdata, [1.0, 2.0]) + 0.01 * randn(rng, length(xdata))
    p0 = [0.5, 0.5]

    # AutoFiniteDiff and AutoForwardDiff are re-exported — no `using ADTypes` needed
    @test AutoFiniteDiff isa Type
    @test AutoForwardDiff isa Type

    # all AutoFiniteDiff fdjtype variants and AutoForwardDiff work as direct kwargs
    for ad in (
        AutoFiniteDiff(),
        AutoFiniteDiff(fdjtype = Val(:central)),
        AutoFiniteDiff(fdjtype = Val(:forward)),
        AutoFiniteDiff(fdjtype = Val(:complex)),
        AutoForwardDiff(),
    )
        fit = curve_fit(model, xdata, ydata, p0; autodiff = ad)
        @test fit.converged
        @test norm(fit.param - [1.0, 2.0]) < 0.05
    end

    # legacy symbols: _autodiff_adtype emits a deprecation warning and maps correctly
    @test_logs (:warn, r"deprecated") LsqFit._autodiff_adtype(:finite)
    @test LsqFit._autodiff_adtype(:finite) == AutoFiniteDiff(fdjtype = Val(:central))
    @test LsqFit._autodiff_adtype(:central) == AutoFiniteDiff(fdjtype = Val(:central))
    @test LsqFit._autodiff_adtype(:finiteforward) ==
          AutoFiniteDiff(fdjtype = Val(:forward))
    @test LsqFit._autodiff_adtype(:finitecomplex) ==
          AutoFiniteDiff(fdjtype = Val(:complex))
    @test LsqFit._autodiff_adtype(:forward) == AutoForwardDiff()
    @test LsqFit._autodiff_adtype(:forwarddiff) == AutoForwardDiff()

    # legacy symbols still produce a correct fit end-to-end
    for sym in (:finite, :central, :finiteforward, :finitecomplex, :forward, :forwarddiff)
        fit = curve_fit(model, xdata, ydata, p0; autodiff = sym)
        @test fit.converged
        @test norm(fit.param - [1.0, 2.0]) < 0.1
    end

    # invalid symbol throws ArgumentError
    @test_throws ArgumentError LsqFit._autodiff_adtype(:bad_symbol)

    # concretely-typed model with AutoForwardDiff gives a friendly, actionable error
    model_typed(x::AbstractVector{Float64}, p::AbstractVector{Float64}) =
        p[1] .* exp.(-x .* p[2])
    err = try
        curve_fit(model_typed, collect(xdata), ydata, p0; autodiff = AutoForwardDiff())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("AutoForwardDiff", err.msg)
    @test occursin("AutoFiniteDiff", err.msg)
    @test occursin("AbstractVector{<:Real}", err.msg)
end

@testset "#167" begin
    x = collect(1:10)
    y = copy(x)
    @. model(x, p) = p[1] * x + p[2]
    p0 = [0.0, -5.0]
    fit = curve_fit(model, x, y, p0) # no bounds
    fit_bounded = curve_fit(model, x, y, p0; upper = [+Inf, -5.0]) # with bounds
    @test coef(fit)[1] < coef(fit_bounded)[1]
    @test coef(fit)[1] ≈ 1
    @test coef(fit_bounded)[1] ≈ 1.22727271
end


@testset "complex" begin
    x = collect(1.0:10)
    @. model(x, p) = p[1] * x + 1im*p[2]
    p = [-2.0, 3]
    y = model(x, p)
    fit = curve_fit(model, x, y, [0.0, -5.0])
    @test fit.converged
    @test fit.param ≈ p atol=1e-9

    x = range(-10, 10, length=101)
    @. model(x, p) = p[1] / (1.0 + 1im * p[3] * (x - p[2]))
    p = [2.0, 3, 0.5]
    y = model(x, p)
    fit = curve_fit(model, x, y, [1.0, -5.0, 0.05];
                    lower=[0.0, -Inf, 0.0])
    @test fit.converged
    @test fit.param ≈ p atol=1e-9
end
