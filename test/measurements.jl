using LsqFit
using Measurements

@testset "MeasurementsExt" begin
    model(x, p) = p[1] .* exp.(-x .* p[2])
    ptrue = [10, 0.3]
    x = LinRange(0, 2, 50)
    y0 = model(x, ptrue)
    σ = rand(1:5, 50)
    y = y0 .± σ
    wt = σ .^ -2
    fit0 = curve_fit(model, x, y0, wt, ptrue) # fit to data using weights
    fit1 = curve_fit(model, x, y, ptrue) # fit to data using Measurements
    @test coef(fit0) ≈ coef(fit1)
end