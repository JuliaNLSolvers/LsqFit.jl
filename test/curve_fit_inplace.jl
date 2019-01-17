struct Jacobian{T}
   J::T
end
Jacobian(x, p) = Jacobian(zeros(eltype(x), length(x), length(p)))

function (lj::Jacobian)(x,p)
    J::Array{Float64,2} = lj.J
    @. J[:,1] = exp(-x*p[2])
    @. @views J[:,2] = -x*p[1]*J[:,1] #views make so that we dont allocate some memory for the J[:,1] slice, it makes the results of the tests much clearer
    J
end
function f!_from_f(f, F::AbstractArray)
    return function ff!(F, x)
            copyto!(F, f(x))

    end
end
let
    # fitting noisy data to an exponential model
    # TODO: Change to `.-x` when 0.5 support is dropped
    model(x, p) = @. p[1] * exp(-x * p[2])
    model_inplace(F, x, p) = (@. F = p[1] * exp(-x * p[2]))

    # some example data
    Random.seed!(12345)
    xdata = range(0, stop=10, length=500000)
    ydata = model(xdata, [1.0, 2.0]) + 0.01*randn(length(xdata))
    p0 = [0.5, 0.5]

    # if your model is differentiable, it can be faster and/or more accurate
    # to supply your own jacobian instead of using the finite difference
    function jacobian_model(x,p)
        J = Array{Float64}(undef, length(x), length(p))
        @. J[:,1] = exp(-x*p[2])     #dmodel/dp[1]
        @. @views J[:,2] = -x*p[1]*J[:,1] 
        J
    end

    function jacobian_model_inplace(J::Array{Float64,2},x,p)
        @. J[:,1] = exp(-x*p[2])     #dmodel/dp[1]
        @. @views J[:,2] = -x*p[1]*J[:,1] 
    end

    jacobian_type = Jacobian(xdata, p0)

    f(p) = model(xdata, p) - ydata
    g(p) = jacobian_model(xdata, p)
    df = OnceDifferentiable(f, g, p0, similar(ydata); inplace = false);
    evalf(x) = NLSolversBase.value!!(df, x)
    evalg(x) = NLSolversBase.jacobian!!(df, x)
    r = evalf(p0);
    j = evalg(p0);

    f_inplace = (F,p) -> (model_inplace(F, xdata, p); @. F = F - ydata)
    g_inplace = (G,p) -> jacobian_model_inplace(G, xdata, p)
    df_inplace = OnceDifferentiable(f_inplace, g_inplace, p0, similar(ydata); inplace = true);
    evalf_inplace(x) = NLSolversBase.value!!(df_inplace,x)
    evalg_inplace(x) = NLSolversBase.jacobian!!(df_inplace,x)
    r_inplace = evalf_inplace(p0);
    j_inplace = evalg_inplace(p0);

    f_type(p) = model(xdata, p) - ydata
    df_type = OnceDifferentiable(f_type, p -> jacobian_type(xdata, p), p0, similar(ydata); inplace = false);
    evalf_type(x) = NLSolversBase.value!!(df_type,x)
    evalg_type(x) = NLSolversBase.jacobian!!(df_type,x)
    r_type = evalf_type(p0);
    j_type = evalg_type(p0);

    @test r == r_inplace == r_type
    @test j == j_inplace == j_type

    println("--------------\nPerformance of non-inplace")
    println("\t Evaluation function")

    stop = 8 #8 because the tests afterwards will call the eval function 8 or 9 times, so it makes it easy to compare
    step = 1

    @time for i=range(0,stop=stop,step=step)
        evalf(p0);
    end

    println("\t Jacobian function")
    @time for i=range(0,stop=stop,step=step)
        evalg(p0);
    end

    println("--------------\nPerformance of inplace")
    println("\t Evaluation function")
    @time for i=range(0,stop=stop,step=step)
        evalf_inplace(p0);
    end

    println("\t Jacobian function")
    @time for i=range(0,stop=stop,step=step)
        evalg_inplace(p0);
    end

    println("--------------\nPerformance of callable type")
    println("\t Evaluation function")
    @time for i=range(0,stop=stop,step=step)
        evalf_type(p0);
    end

    println("\t Jacobian function")
    @time for i=range(0,stop=stop,step=step)
        evalg_type(p0);
    end


    curve_fit(model, jacobian_model, xdata, ydata, p0); #warmup
    curve_fit(model_inplace, jacobian_model_inplace, xdata, ydata, p0; inplacejac = true, inplacef = true);
    curve_fit(model, (x,p)-> jacobian_type(x,p), xdata, ydata, p0);

    println("--------------\nPerformance of curve_fit")

    println("\t Non-inplace")
    jacobian_fit = @time curve_fit(model, jacobian_model, xdata, ydata, p0; maxIter=100)
    @test jacobian_fit.converged

    println("\t Inplace")
    jacobian_fit_inplace = @time curve_fit(model_inplace, jacobian_model_inplace, xdata, ydata, p0; inplacejac = true, inplacef = true, maxIter=100)
    @test jacobian_fit_inplace.converged

    println("\t Callable type")
    jacobian_fit_type = @time curve_fit(model, (x,p)-> jacobian_type(x,p), xdata, ydata, p0; maxIter=100)
    @test jacobian_fit_type.converged

    @test jacobian_fit.param == jacobian_fit_inplace.param == jacobian_fit_type.param
    # some example data
    yvars = 1e-6*rand(length(xdata))
    ydata = model(xdata, [1.0, 2.0]) + sqrt.(yvars) .* randn(length(xdata))

    println("--------------\nPerformance of curve_fit with weights")

    curve_fit(model, jacobian_model, xdata, ydata, 1 ./ yvars, [0.5, 0.5]);
    curve_fit(model_inplace, jacobian_model_inplace, xdata, ydata, 1 ./ yvars, [0.5, 0.5]; inplacejac = true, inplacef = true, maxIter=100);
    curve_fit(model, (x,p)-> jacobian_type(x,p), xdata, ydata, 1 ./ yvars, [0.5, 0.5]);

    println("\t Non-inplace with weights")
    fit = @time curve_fit(model, jacobian_model, xdata, ydata, 1 ./ yvars, [0.5, 0.5]; maxIter=100)
    @test fit.converged

    println("\t Inplace with weights")
    fit_inplace = @time curve_fit(model_inplace, jacobian_model_inplace, xdata, ydata, 1 ./ yvars, [0.5, 0.5]; inplacejac = true, inplacef = true, maxIter=100)
    @test fit_inplace.converged

    println("\t Callable type with weights")
    fit_type = @time curve_fit(model, (x,p)-> jacobian_type(x,p), xdata, ydata, 1 ./ yvars, [0.5, 0.5]; maxIter=100)
    @test fit_type.converged

    @test fit.param == fit_inplace.param == fit_type.param
end
