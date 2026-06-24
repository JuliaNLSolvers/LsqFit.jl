using LsqFit, Test, LinearAlgebra, StableRNGs

@testset "typed weights (#255)" begin
    model(x, p) = p[1] .* exp.(p[2] .* x)

    # dataset from issue #255
    x = Float64[1, 2, 4, 5, 8]
    y = Float64[3, 4, 6, 11, 20]
    σ = 0.05 .* y
    wt = 1 ./ σ .^ 2
    p0 = [2.0, 0.3]

    fit_known = curve_fit(model, x, y, PrecisionWeights(wt), p0) # known variance
    fit_rel = curve_fit(model, x, y, AnalyticWeights(wt), p0)   # estimate scale

    @testset "point estimates agree, std errors differ" begin
        @test coef(fit_known) ≈ coef(fit_rel) rtol = 1e-8
        @test coef(fit_known) ≈ [2.263, 0.275] rtol = 1e-2
        # PrecisionWeights reproduce the (small) known-variance errors
        @test stderror(fit_known) ≈ [0.0968, 0.0091] rtol = 1e-2
        # AnalyticWeights reproduce the Origin/LabPlot/mycurvefit errors
        @test stderror(fit_rel) ≈ [0.2579, 0.0243] rtol = 1e-2
        # the AnalyticWeights errors are larger (scale is estimated)
        @test all(stderror(fit_rel) .> stderror(fit_known))
    end

    @testset "AnalyticWeights are scale-invariant" begin
        se = stderror(curve_fit(model, x, y, AnalyticWeights(wt), p0))
        se10 = stderror(curve_fit(model, x, y, AnalyticWeights(10 .* wt), p0))
        @test se ≈ se10 rtol = 1e-6
        # uniform AnalyticWeights == unweighted; PrecisionWeights of ones do not
        unwt = curve_fit(model, x, y, p0)
        a1 = curve_fit(model, x, y, AnalyticWeights(ones(length(x))), p0)
        @test stderror(a1) ≈ stderror(unwt) rtol = 1e-6
        @test !isapprox(
            stderror(curve_fit(model, x, y, PrecisionWeights(ones(length(x))), p0)),
            stderror(unwt),
        )
    end

    @testset "vcov formulas" begin
        J = fit_known.jacobian
        @test vcov(fit_known) ≈ inv(J' * J) rtol = 1e-8                 # known: no MSE
        @test vcov(fit_rel) ≈ inv(J' * J) * LsqFit.mse(fit_rel) rtol = 1e-8
        @test stderror(fit_known) ≈ sqrt.(diag(vcov(fit_known)))
    end

    @testset "bare vector/matrix weights are rejected" begin
        # Bare weights were removed in 1.0; they must be wrapped in a weight type.
        # Cover all four weighted methods: vector/matrix × with/without a Jacobian.
        jac(x, p) = hcat(exp.(p[2] .* x), p[1] .* x .* exp.(p[2] .* x))
        W = LinearAlgebra.diagm(wt)
        @test_throws ArgumentError curve_fit(model, x, y, wt, p0)
        @test_throws ArgumentError curve_fit(model, x, y, W, p0)
        @test_throws ArgumentError curve_fit(model, jac, x, y, wt, p0)
        @test_throws ArgumentError curve_fit(model, jac, x, y, W, p0)
    end

    @testset "FrequencyWeights count observations" begin
        counts = [3, 1, 1, 2, 1]
        fit_freq = curve_fit(model, x, y, FrequencyWeights(counts), p0)
        @test nobs(fit_freq) == sum(counts)
        @test dof(fit_freq) == sum(counts) - length(p0)
    end

    @testset "unsupported weight types error" begin
        @test_throws ArgumentError vcov(curve_fit(model, x, y, ProbabilityWeights(wt), p0))
        @test_throws ArgumentError vcov(curve_fit(model, x, y, Weights(wt), p0))
    end

    @testset "Monte-Carlo coverage" begin
        truth = [2.0, 0.25]
        xg = collect(range(0.5, 6; length = 15))
        rng = StableRNG(20240617)
        N = 3000
        hitA = zeros(2)
        hitV = zeros(2)
        for _ = 1:N
            sd = 0.10 .* model(xg, truth)
            yg = model(xg, truth) .+ sd .* randn(rng, length(xg))
            w = 1 ./ sd .^ 2
            fA = curve_fit(model, xg, yg, AnalyticWeights(w), copy(truth))  # estimate scale
            fV = curve_fit(model, xg, yg, PrecisionWeights(w), copy(truth)) # known variance
            for (k, (lo, hi)) in enumerate(confint(fA; level = 0.95))
                hitA[k] += lo <= truth[k] <= hi
            end
            for (k, (lo, hi)) in enumerate(confint(fV; level = 0.95))
                hitV[k] += lo <= truth[k] <= hi
            end
        end
        # AnalyticWeights (estimate scale, Student-t) and PrecisionWeights (known
        # variance, normal) should both be close to the nominal 95%, each with the
        # reference distribution matching its assumption.
        @test all(abs.(100 .* hitA ./ N .- 95) .< 2)
        @test all(abs.(100 .* hitV ./ N .- 95) .< 2.5)
    end
end
