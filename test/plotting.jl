let
    @testset "Plotting" begin
        model(x, p) = @. p[1] * x^2 + p[2] * x + p[3] + p[4] * exp(-(x - p[5])^2 / p[6]^2)
        p_true = [0.5, 0.7, 0.0, 4.5, 0.3, 0.5]
        xdata = -4.0:0.5:4.0
        ydata = model(xdata, p_true) + 0.7 * randn(size(xdata))

        f = curve_fit(model, xdata, ydata, fill(1.0, 6))

        p = plot(
            map((:neither, :prediction, :confidence, :both)) do purpose
                plot(
                    xdata,
                    ydata;
                    seriestype=:scatter,
                    label="Data",
                    legend_foreground_color=nothing,
                    legend_background_color=nothing,
                    legend=:topleft,
                )
                plot!(
                    x -> model(x, p_true);
                    seriestype=:line,
                    label="Ground truth",
                    linestyle=:dot,
                )
                plot!(model, f; purpose=purpose, title="$purpose")
            end...;
            layout=(2, 2),
        )
        @test p isa Plots.Plot
        savefig(p, "plots.png")
        println("Plots saved to `test/plots.png`")
    end
end
