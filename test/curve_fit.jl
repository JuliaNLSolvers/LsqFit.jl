using LsqFit, Test, StableRNGs, LinearAlgebra
@testset "curve fit" begin
    # before testing the model, check whether missing/null data is rejected
    tdata = [rand(1:10, 5)..., missing]
    @test_throws ErrorException(
        "The independent variable (`x`) contains `missing` values and a fit cannot be performed",
    ) LsqFit.check_data_health(tdata, tdata)
    # missing in `y` and in the weights are rejected too
    cleandata = collect(1:6)
    @test_throws ErrorException(
        "The dependent variable (`y`) contains `missing` values and a fit cannot be performed",
    ) LsqFit.check_data_health(cleandata, tdata)
    @test_throws ErrorException(
        "Weight data contains `missing` values and a fit cannot be performed",
    ) LsqFit.check_data_health(cleandata, cleandata, tdata)
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

        fit = curve_fit(model, xdata, ydata, PrecisionWeights(1 ./ yvars), T.([0.5, 0.5]))

        @test norm(fit.param - [1.0, 2.0]) < 0.05
        @test fit.converged

        # test matrix valued weights ( #161 )
        weights = PrecisionMatrix(LinearAlgebra.diagm(1 ./ yvars))
        fit_matrixweights = curve_fit(model, xdata, ydata, weights, T.([0.5, 0.5]))
        @test fit.param == fit_matrixweights.param

        # can also get error estimates on the fit parameters
        errors = margin_error(fit, T(0.1))

        @test norm(errors - [0.017, 0.075]) < 0.1

        # test with user-supplied jacobian and weights
        fit = curve_fit(model, jacobian_model, xdata, ydata, PrecisionWeights(1 ./ yvars), p0)
        println("norm(fit.param - [1.0, 2.0]) < 0.05 ? ", norm(fit.param - [1.0, 2.0]))
        @test norm(fit.param - [1.0, 2.0]) < 0.05
        @test fit.converged

        # Parameters can also be inferred using arbitrary precision
        fit = curve_fit(
            model,
            xdata,
            ydata,
            PrecisionWeights(1 ./ yvars),
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
            PrecisionWeights(1 ./ yvars),
            BigFloat.(p0);
            x_tol = T(1e-20),
            g_tol = T(1e-20),
        )
        @test fit.converged

        curve_fit(model, jacobian_model, xdata, ydata, PrecisionWeights(1 ./ yvars), [0.5, 0.5]; tau = 0.0001)
    end

end

# Covariance of weighted fits, see #255. The weights are folded into the
# Jacobian, so they are treated as the known inverse variance and the weighted
# covariance is *not* scaled by the mean squared error. This documents the
# intentional difference from the unweighted case and guards the QR refactor.
@testset "weighted vcov (#255)" begin
    model(x, p) = p[1] .* exp.(-x .* p[2])

    rng = StableRNG(125)
    xdata = range(0, stop = 10, length = 20)
    σ = 0.05
    ydata = model(xdata, [1.0, 2.0]) + σ * randn(rng, length(xdata))
    p0 = [0.5, 0.5]

    wt = fill(1 / σ^2, length(xdata))
    fit = curve_fit(model, xdata, ydata, PrecisionWeights(wt), p0)

    # QR-based covariance equals the algebraic inv(J'J); J already carries the
    # weights, so this is the known-inverse-variance covariance with no MSE.
    J = fit.jacobian
    @test vcov(fit) ≈ inv(J' * J)
    @test vcov(fit) ≉ inv(J' * J) * LsqFit.mse(fit)

    # Vector weights and the equivalent diagonal matrix weights agree.
    fit_mat = curve_fit(model, xdata, ydata, PrecisionMatrix(LinearAlgebra.diagm(wt)), p0)
    @test vcov(fit) ≈ vcov(fit_mat)

    # Weights of one give the same parameter estimates as an unweighted fit,
    # but a different covariance: the unweighted case additionally estimates
    # the residual variance via the MSE (#255).
    fit_ones = curve_fit(model, xdata, ydata, PrecisionWeights(ones(length(xdata))), p0)
    fit_unwt = curve_fit(model, xdata, ydata, p0)
    @test fit_ones.param ≈ fit_unwt.param
    @test vcov(fit_ones) ≈ vcov(fit_unwt) / LsqFit.mse(fit_unwt)
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

    # Symbol-based autodiff was removed in 1.0; only ADTypes backends are accepted.
    @test !isdefined(LsqFit, :_autodiff_adtype)
    for sym in (:finite, :central, :forward, :forwarddiff)
        @test_throws Exception curve_fit(model, xdata, ydata, p0; autodiff = sym)
    end

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

@testset "non-Dual errors from the optimizer are rethrown" begin
    # A user-supplied Jacobian that throws something other than a Dual-related
    # MethodError must propagate unchanged (not be reinterpreted as an autodiff
    # incompatibility).
    model(x, p) = p[1] .* exp.(-x .* p[2])
    x = collect(range(0, 10, length = 20))
    y = model(x, [1.0, 2.0])
    badjac(x, p) = throw(ArgumentError("boom in jacobian"))
    @test_throws ArgumentError("boom in jacobian") curve_fit(model, badjac, x, y, [0.5, 0.5])
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

@testset "Pointwise models (#145)" begin
    mp(x, p) = p[1] * exp(p[2] * x)
    mv(x, p) = p[1] .* exp.(p[2] .* x)
    mv!(F, x, p) = (@. F = p[1] * exp(p[2] * x))
    gp(x, p) = [exp(p[2] * x), p[1] * x * exp(p[2] * x)]
    jv(x, p) = hcat(exp.(p[2] .* x), p[1] .* x .* exp.(p[2] .* x))
    jv!(J, x, p) = (J .= jv(x, p))

    x = Float64[1, 2, 4, 5, 8]
    y = Float64[3, 4, 6, 11, 20]
    p0 = [2.0, 0.3]
    w = 1 ./ (0.05 .* y) .^ 2

    @testset "match the vectorised fit" begin
        for wt in ((), (PrecisionWeights(w),)), inplace in (false, true)
            vm = inplace ? mv! : mv
            vj = inplace ? jv! : jv
            ref = curve_fit(vm, x, y, wt..., p0; inplace = inplace)
            fit = curve_fit(Pointwise(mp), x, y, wt..., p0; inplace = inplace)
            @test coef(fit) ≈ coef(ref) rtol = 1e-12
            ref_jac = curve_fit(vm, vj, x, y, wt..., p0; inplace = inplace)
            fit_jac = curve_fit(
                Pointwise(mp),
                PointwiseJacobian(gp),
                x,
                y,
                wt...,
                p0;
                inplace = inplace,
            )
            @test coef(fit_jac) ≈ coef(ref_jac) rtol = 1e-12
            @test fit_jac.jacobian ≈ ref_jac.jacobian rtol = 1e-12
        end
        W = PrecisionMatrix(LinearAlgebra.diagm(w))
        @test coef(curve_fit(Pointwise(mp), x, y, W, p0)) ≈ coef(curve_fit(mv, x, y, W, p0)) rtol =
            1e-12
        @test coef(curve_fit(Pointwise(mp), PointwiseJacobian(gp), x, y, W, p0)) ≈
              coef(curve_fit(mv, jv, x, y, W, p0)) rtol = 1e-12
        # the analytic gradient agrees with the autodiff Jacobian at the solution
        fit = curve_fit(Pointwise(mp), x, y, p0)
        @test PointwiseJacobian(gp)(x, coef(fit)) ≈ fit.jacobian
        @test Pointwise(mp)(x, coef(fit)) ≈ y + residuals(fit)
    end

    @testset "gradient forms" begin
        p = [1.5, 0.2]
        J = jv(x, p)
        @test PointwiseJacobian(gp)(x, p) ≈ J
        @test PointwiseJacobian((x, p) -> Tuple(gp(x, p)))(x, p) ≈ J
        @test PointwiseJacobian((x, p) -> permutedims(gp(x, p)))(x, p) ≈ J
        # one parameter: the gradient is a number
        @test PointwiseJacobian((x, p) -> x * exp(p[1] * x))(x, [0.2]) ≈
              reshape(x .* exp.(0.2 .* x), :, 1)
        fit_tuple = curve_fit(
            Pointwise(mp),
            PointwiseJacobian((x, p) -> Tuple(gp(x, p))),
            x,
            y,
            p0;
            inplace = true,
        )
        @test coef(fit_tuple) ≈ coef(curve_fit(mv!, jv!, x, y, p0; inplace = true)) rtol =
            1e-12
    end

    @testset "several independent variables" begin
        X = [1.0 0.5; 2.0 1.0; 3.0 0.2; 4.0 2.0; 5.0 1.5; 6.0 0.7]
        mv2(X, p) = p[1] .* exp.(p[2] .* X[:, 1] .+ p[3] .* X[:, 2])
        mp2(xᵢ, p) = p[1] * exp(p[2] * xᵢ[1] + p[3] * xᵢ[2])
        y2 = mv2(X, [1.0, 0.3, -0.5]) .+ [0.01, -0.02, 0.0, 0.03, -0.01, 0.02]
        q0 = [0.5, 0.1, 0.1]
        ref = curve_fit(mv2, X, y2, q0)
        # a matrix has one observation per row; a vector holds one per element
        for xdata in (X, Tuple.(eachrow(X)), collect.(eachrow(X))), inplace in (false, true)
            fit = curve_fit(Pointwise(mp2), xdata, y2, q0; inplace = inplace)
            @test coef(fit) ≈ coef(ref) rtol = 1e-12
        end
        @test_throws ErrorException LsqFit.check_data_health([(1.0, NaN), (2.0, 1.0)], y2)
        @test_throws ErrorException LsqFit.check_data_health([[1.0, missing]], y2)
    end

    @testset "errors" begin
        x_long = [x; 9.0]
        @test_throws DimensionMismatch curve_fit(Pointwise(mp), x_long, y, p0)
        @test_throws DimensionMismatch curve_fit(Pointwise(mp), x_long, y, p0; inplace = true)
        @test_throws DimensionMismatch curve_fit(
            Pointwise(mp),
            PointwiseJacobian(gp),
            x_long,
            y,
            p0;
            inplace = true,
        )
        @test_throws DimensionMismatch curve_fit(
            Pointwise(mp),
            PointwiseJacobian((x, p) -> [1.0, 2.0, 3.0]),
            x,
            y,
            p0,
        )
        @test_throws ArgumentError curve_fit(Pointwise((x, p) -> [mp(x, p)]), x, y, p0)
        @test_throws ArgumentError curve_fit(Pointwise(mp), reshape(x, 1, 1, :), y, p0)
    end
end
