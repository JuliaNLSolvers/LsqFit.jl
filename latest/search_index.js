var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#LsqFit.jl-1",
    "page": "Home",
    "title": "LsqFit.jl",
    "category": "section",
    "text": "Basic least-squares fitting in pure Julia under an MIT license.The basic functionality was originaly in Optim.jl, before being separated into this library.  At this time, LsqFit only utilizes the Levenberg-Marquardt algorithm for non-linear fitting.LsqFit.jl is part of the JuliaNLSolvers family.Source Package Evaluator Build Status\n(Image: Source) (Image: LsqFit) (Image: LsqFit) (Image: LsqFit) (Image: LsqFit) (Image: Build Status)"
},

{
    "location": "#Install-1",
    "page": "Home",
    "title": "Install",
    "category": "section",
    "text": "To install the package, runPkg.add(\"LsqFit\")If you want the latest features, also runPkg.checkout(\"LsqFit\")To use the package in your codejulia> using LsqFit"
},

{
    "location": "getting_started/#",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "page",
    "text": ""
},

{
    "location": "getting_started/#Getting-Started-1",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "section",
    "text": "First, import the package.julia> using LsqFitDefine a two-parameter exponential decay model, where t is a one-element independent variable, p_1 and p_2 are parameters.The model function is:m(t boldsymbolp) = p_1 exp(-p_2 t)julia> # t: array of independent variable\njulia> # p: array of model parameters\njulia> model(t, p) = p[1] * exp.(-p[2] * t)For illustration purpose, we generate some fake data.julia> # tdata: data of independent variable\njulia> # ydata: data of dependent variable\njulia> tdata = linspace(0,10,20)\njulia> ydata = model(tdata, [1.0 2.0]) + 0.01*randn(length(tdata))Before fitting the data, we also need a initial value of parameters for curve_fit().julia> p0 = [0.5, 0.5]Run curve_fit() to fit the data and get the estimated parameters.julia> fit = curve_fit(model, tdata, ydata, p0)\njulia> param = fit.param\n2-element Array{Float64,1}:\n 1.01105\n 2.0735LsqFit.jl also provides functions to exam the goodness of fit. estimate_covar(fit) computes the estimated covariance matrix.julia> cov = estimate_covar(fit)\n2×2 Array{Float64,2}:\n 0.000116545  0.000174633\n 0.000174633  0.00258261standard_error(fit) returns the standard error of each parameter.julia> se = standard_error(fit)\n2-element Array{Float64,1}:\n 0.0107956\n 0.0508193To get the confidence interval at 10% significance level, run confidence_interval(fit, alpha), which essentially computes the estimate parameter value ± (standard error * critical value from t-distribution).julia> confidence_interval = confidence_interval(fit, 0.1)\n2-element Array{Tuple{Float64,Float64},1}:\n (0.992333, 1.02977)\n (1.98537, 2.16162)For more details of LsqFit.jl, check Tutorial and API References section."
},

{
    "location": "tutorial/#",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "page",
    "text": ""
},

{
    "location": "tutorial/#Tutorial-1",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "section",
    "text": ""
},

{
    "location": "tutorial/#Introduction-to-Nonlinear-Regression-1",
    "page": "Tutorial",
    "title": "Introduction to Nonlinear Regression",
    "category": "section",
    "text": "Assume that, for the ith observation, the relationship between independent variable mathbfx_i=beginbmatrix x_1i x_2i ldots x_pi endbmatrix and dependent variable Y_i follows:Y_i = m(mathbfx_i boldsymbolgamma) + epsilon_iwhere m is a non-linear model function depends on the independent variable mathbfx_i and the parameter vector boldsymbolgamma. In order to find the parameter boldsymbolgamma that \"best\" fit our data, we choose the parameter boldsymbolgamma which minimizes the sum of squared residuals from our data, i.e. solves the problem:undersetboldsymbolgammamathrmmin quad s(boldsymbolgamma)= sum_i=1^n m(mathbfx_i boldsymbolgamma) - y_i^2Given that the function m is non-linear, there\'s no analytical solution for the best boldsymbolgamma. We have to use computational tools, which is LsqFit.jl in this tutorial, to find the least squares solution.One example of non-linear model is the exponential model, which takes a one-element predictor variable t. The model function is:m(t boldsymbolgamma) = gamma_1 exp(gamma_2 t)and the model becomes:Y_i = gamma_1 exp(gamma_2 t_i) + epsilon_iTo fit data using LsqFit.jl, pass the defined model function (m), data (tdata and ydata) and the initial parameter value (p0) to curve_fit(). For now, LsqFit.jl only supports the Levenberg Marquardt algorithm.julia> # t: array of independent variables\njulia> # p: array of model parameters\njulia> m(t, p) = p[1] * exp.(p[2] * t)\njulia> p0 = [0.5, 0.5]\njulia> fit = curve_fit(m, tdata, ydata, p0)It will return a composite type LsqFitResult, with some interesting values:*	fit.dof: degrees of freedom *	fit.param: best fit parameters *	fit.resid: vector of residuals *	fit.jacobian: estimated Jacobian at the solution"
},

{
    "location": "tutorial/#Jacobian-Calculation-1",
    "page": "Tutorial",
    "title": "Jacobian Calculation",
    "category": "section",
    "text": "The Jacobian J_f(mathbfx) of a vector function f(mathbfx) mathbbR_m to mathbbR_n is deﬁned as the matrix with elements:J_f(mathbfx)_ij = fracpartial f_i(mathbfx)partial x_jThe matrix is therefore:J_f(mathbfx) = beginbmatrix\n                fracpartial f_1partial x_1fracpartial f_1partial x_2dotsfracpartial f_1partial x_m\n                fracpartial f_2partial x_1fracpartial f_2partial x_2dotsfracpartial f_2partial x_m\n                vdotsvdotsddotsvdots\n                fracpartial f_npartial x_1fracpartial f_npartial x_2dotsfracpartial f_npartial x_m\n                endbmatrixThe Jacobian of the exponential model function with respect to boldsymbolgamma is:J_m(t boldsymbolgamma) = beginbmatrix\n            fracpartial mpartial gamma_1 \n            fracpartial mpartial gamma_2 \n            endbmatrix\n          = beginbmatrix\n            exp(gamma_2 t) \n            t gamma_1 exp(gamma_2 t) \n            endbmatrixBy default, the finite difference method, Calculus.jacobian(), is used to approximate the Jacobian for the data fitting algorithm and covariance computation. Alternatively, a function which calculates the Jacobian can be supplied to curve_fit() for faster and/or more accurate results.function j_m(t,p)\n    J = Array{Float64}(length(t),length(p))\n    J[:,1] = exp.(p[2] .* t)       #df/dp[1]\n    J[:,2] = t .* p[1] .* J[:,1]   #df/dp[2]\n    J\nend\n\nfit = curve_fit(m, j_m, tdata, ydata, p0)"
},

{
    "location": "tutorial/#Linear-Approximation-1",
    "page": "Tutorial",
    "title": "Linear Approximation",
    "category": "section",
    "text": "The non-linear function m can be approximated as a linear function by Talor expansion:m(mathbfx_i boldsymbolgamma+boldsymbolh) approx m(mathbfx_i boldsymbolgamma) +  nabla m(mathbfx_i boldsymbolgamma)boldsymbolhwhere boldsymbolgamma is a fixed vector, boldsymbolh is a very small-valued vector and nabla m(mathbfx_i boldsymbolgamma) is the gradient at mathbfx_i.Consider the residual vector functon r(boldsymbolgamma)=beginbmatrix                           r_1(boldsymbolgamma) \n                          r_2(boldsymbolgamma) \n                          vdots\n                          r_n(boldsymbolgamma)                           endbmatrix with entries:r_i(boldsymbolgamma) = m(mathbfx_i boldsymbolgamma) - Y_iEach entry\'s linear approximation can hence be written as:beginalign\nr_i(boldsymbolgamma+boldsymbolh) = m(mathbfx_i boldsymbolgamma+boldsymbolh) - Y_i\napprox m(mathbfx_i boldsymbolgamma) + nabla m(mathbfx_i boldsymbolgamma)h - Y_i\n= r_i(boldsymbolgamma) + nabla m(mathbfx_i boldsymbolgamma)h\nendalignSince the ith row of J(boldsymbolgamma) equals the transpose of the gradient of m(mathbfx_i boldsymbolgamma), the vector function r(boldsymbolgamma+boldsymbolh) can be approximated as:r(boldsymbolgamma+boldsymbolh) approx r(boldsymbolgamma) + J(boldsymbolgamma)hwhich is a linear function on boldsymbolh since boldsymbolgamma is a fixed vector."
},

{
    "location": "tutorial/#Goodness-of-Fit-1",
    "page": "Tutorial",
    "title": "Goodness of Fit",
    "category": "section",
    "text": "The linear approximation of the non-linear least squares problem leads to the approximation of the covariance matrix of each parameter, from which we can perform regression analysis.Consider a least squares solution boldsymbolgamma^*, which is a local minimizer of the non-linear problem:boldsymbolgamma^* = undersetboldsymbolgammamathrmargmin  sum_i=1^n m(mathbfx_i boldsymbolgamma) - y_i^2Set boldsymbolgamma^* as the fixed point in linear approximation, r(boldsymbolgamma^*) = r and J(boldsymbolgamma^*) = J. A parameter vector near boldsymbolgamma^* can be expressed as boldsymbolgamma=boldsymbolgamma^* + h. The local approximation for the least squares problem is:undersetboldsymbolgammamathrmmin quad s(boldsymbolgamma)=s(boldsymbolgamma^*+boldsymbolh) approx Jh + rJh + rwhich is essentially the linear least squares problem:undersetboldsymbolbetamathrmmin quad Xbeta-YXbeta-Ywhere X=J, beta=boldsymbolh and Y=-r(boldsymbolgamma). Solve the equation where the partial derivatives equal to 0, the analytical solution is:hatboldsymbolh=hatboldsymbolgamma-boldsymbolgamma^*approx-JJ^-1JrThe covariance matrix for the analytical solution is:mathbfCov(hatboldsymbolgamma) = mathbfCov(boldsymbolh) = JJ^-1JmathbfE(rr)JJJ^-1Note that r is the residual vector at the best fit point boldsymbolgamma^*, with entries r_i = Y_i - m(mathbfx_i boldsymbolgamma^*)=epsilon_i. hatboldsymbolgamma is very close to boldsymbolgamma^* and therefore can be replaced by boldsymbolgamma^*.mathbfCov(boldsymbolgamma^*) approx mathbfCov(hatboldsymbolgamma)Assume the errors in each sample are independent, normal distributed with zero mean and same variance, i.e. epsilon sim N(0 sigma^2I), the covariance matrix from the linear approximation is therefore:mathbfCov(boldsymbolgamma^*) = JJ^-1JmathbfCov(epsilon)JJJ^-1 = sigma^2JJ^-1where sigma^2 could be estimated as residual sum of squares devided by degrees of freedom:hatsigma^2=fracs(boldsymbolgamma^*)n-pIn LsqFit.jl, the covariance matrix calculation uses QR decomposition to be more computationally stable, which has the form:mathbfCov(boldsymbolgamma^*) = hatsigma^2 mathrmR^-1(mathrmR^-1)estimate_covar() computes the covariance matrix of fit:julia> cov = estimate_covar(fit)\n2×2 Array{Float64,2}:\n 0.000116545  0.000174633\n 0.000174633  0.00258261The standard error is then the square root of each diagonal elements of the covariance matrix. standard_error() returns the standard error of each parameter:julia> se = standard_error(fit)\n2-element Array{Float64,1}:\n 0.0114802\n 0.0520416margin_error() computes the product of standard error and the critical value of each parameter at a certain significance level (default is 5%) from t-distribution. The margin of error at 10% significance level can be computed by:julia> margin_of_error = margin_error(fit, 0.1)\n2-element Array{Float64,1}:\n 0.0199073\n 0.0902435confidence_interval() returns the confidence interval of each parameter at certain significance level, which is essentially the estimate value ± margin of error. To get the confidence interval at 10% significance level, run:julia> confidence_intervals = confidence_interval(fit, 0.1)\n2-element Array{Tuple{Float64,Float64},1}:\n (0.976316, 1.01613)\n (1.91047, 2.09096)"
},

{
    "location": "tutorial/#Weighted-Least-Squares-1",
    "page": "Tutorial",
    "title": "Weighted Least Squares",
    "category": "section",
    "text": "curve_fit() also accepts weight parameter (wt) to perform Weighted Least Squares and General Least Squares, where the parameter boldsymbolgamma^* minimizes the weighted residual sum of squares.Weight parameter (wt) is an array or a matrix of weights for each sample. To perform Weighted Least Squares, pass the weight array [w_1, w_2, ..., w_n] or the weight matrix W:mathbfW = beginbmatrix\n    w_1     0       cdots  0\n    0       w_2     cdots  0\n    vdots  vdots  ddots  vdots\n    0       0     cdots  w_n\n    endbmatrixThe weighted least squares problem becomes:undersetboldsymbolgammamathrmmin quad s(boldsymbolgamma)= sum_i=1^n w_im(mathbfx_i boldsymbolgamma) - Y_i^2in matrix form:undersetboldsymbolgammamathrmmin quad s(boldsymbolgamma)= r(boldsymbolgamma)Wr(boldsymbolgamma)where r(boldsymbolgamma)=beginbmatrix                           r_1(boldsymbolgamma) \n                          r_2(boldsymbolgamma) \n                          vdots\n                          r_n(boldsymbolgamma)                           endbmatrix is a residual vector function with entries:r_i(boldsymbolgamma) = m(mathbfx_i boldsymbolgamma) - Y_iThe algorithm in LsqFit.jl will then provide a least squares solution boldsymbolgamma^*.note: Note\nIn LsqFit.jl, the residual function passed to levenberg_marquardt() is in different format, if the weight is a vector:r(p) = sqrt.(wt) .* ( model(xpts, p) - ydata )\nlmfit(r, g, p0, wt; kwargs...)r_i(boldsymbolgamma) = sqrtw_i cdot m(mathbfx_i boldsymbolgamma) - Y_iCholesky decomposition, which is effectively a sqrt of a matrix, will be performed if the weight is a matrix:u = chol(wt)\nr(p) = u * ( model(xpts, p) - ydata )\nlmfit(r, p0, wt; kwargs...)r_i(boldsymbolgamma) = sqrtw_i cdot m(mathbfx_i boldsymbolgamma) - Y_iThe solution will be the same as the least squares problem mentioned in the tutorial.Set r(boldsymbolgamma^*) = r and J(boldsymbolgamma^*) = J, the linear approximation of the weighted least squares problem is then:undersetboldsymbolgammamathrmmin quad s(boldsymbolgamma) = s(boldsymbolgamma^* + boldsymbolh) approx Jboldsymbolh+rWJboldsymbolh+rThe analytical solution to the linear approximation is:hatboldsymbolh=hatboldsymbolgamma-boldsymbolgamma^*approx-JWJ^-1JWrAssume the errors in each sample are independent, normal distributed with zero mean and different variances (heteroskedastic error), i.e. epsilon sim N(0 Sigma), where:Sigma = beginbmatrix\n         sigma_1^2     0       cdots  0\n         0       sigma_2^2     cdots  0\n         vdots  vdots  ddots  vdots\n         0       0     cdots  sigma_n^2\n         endbmatrixWe know the error variance and we set the weight as the inverse of the variance (the optimal weight), i.e. W = Sigma^-1:mathbfW =  beginbmatrix\n              w_1     0       cdots  0\n              0       w_2     cdots  0\n              vdots  vdots  ddots  vdots\n              0       0     cdots  w_n\n              endbmatrix\n           =  beginbmatrix\n               frac1sigma_1^2     0       cdots  0\n               0       frac1sigma_2^2     cdots  0\n               vdots  vdots  ddots  vdots\n               0       0     cdots  frac1sigma_n^2\n               endbmatrixThe covariance matrix is now:Cov(boldsymbolgamma^*) approx  JWJ^-1JW Sigma WJJWJ^-1 = JWJ^-1If we only know the relative ratio of different variances, i.e. epsilon sim N(0 sigma^2W^-1), the covariance matrix will be:mathbfCov(boldsymbolgamma^*) = sigma^2JWJ^-1where sigma^2 is estimated. In this case, if we set W = I, the result will be the same as the unweighted version. However, curve_fit() currently does not support this implementation. curve_fit() assumes the weight as the inverse of the error covariance matrix rather than the ratio of error covariance matrix, i.e. the covariance of the estimated parameter is calculated as covar = inv(J\'*fit.wt*J).note: Note\nPassing vector of ones as the weight vector will cause mistakes in covariance estimation.Pass the vector of 1 ./ var(ε) or the matrix inv(covar(ε)) as the weight parameter (wt) to the function curve_fit():julia> wt = inv(cov_ε)\njulia> fit = curve_fit(m, tdata, ydata, wt, p0)\njulia> cov = estimate_covar(fit)note: Note\nIf the weight matrix is not a diagonal matrix, General Least Squares will be performed."
},

{
    "location": "tutorial/#General-Least-Squares-1",
    "page": "Tutorial",
    "title": "General Least Squares",
    "category": "section",
    "text": "Assume the errors in each sample are correlated, normal distributed with zero mean and different variances (heteroskedastic and autocorrelated error), i.e. epsilon sim N(0 Sigma).Set the weight matrix as the inverse of the error covariance matrix (the optimal weight), i.e. W = Sigma^-1, we will get the parameter covariance matrix:mathbfCov(boldsymbolgamma^*) approx  JWJ^-1JW Sigma WJJWJ^-1 = JWJ^-1Pass the matrix inv(covar(ε)) as the weight parameter (wt) to the function curve_fit():julia> wt = 1 ./ yvar\njulia> fit = curve_fit(m, tdata, ydata, wt, p0)\njulia> cov = estimate_covar(fit)"
},

{
    "location": "tutorial/#Estimate-the-Optimal-Weight-1",
    "page": "Tutorial",
    "title": "Estimate the Optimal Weight",
    "category": "section",
    "text": "In most cases, the variances of errors are unknown. To perform Weighted Least Square, we need estimate the variances of errors first, which is the squared residual of ith sample:widehatmathbfVar(epsilon_i) = widehatmathbfE(epsilon_i epsilon_i) = r_i(boldsymbolgamma^*)Unweighted fitting (OLS) will return the residuals we need, since the estimator of OLS is unbiased. Then pass the reciprocal of the residuals as the estimated optimal weight to perform Weighted Least Squares:julia> fit_OLS = curve_fit(m, tdata, ydata, p0)\njulia> wt = 1 ./ fit_OLS.resid\njulia> fit_WLS = curve_fit(m, tdata, ydata, wt, p0)\njulia> cov = estimate_covar(fit_WLS)"
},

{
    "location": "tutorial/#References-1",
    "page": "Tutorial",
    "title": "References",
    "category": "section",
    "text": "Hansen, P. C., Pereyra, V. and Scherer, G. (2013) Least squares data fitting with applications. Baltimore, Md: Johns Hopkins University Press, p. 147-155.Kutner, M. H. et al. (2005) Applied Linear statistical models.Weisberg, S. (2014) Applied linear regression. Fourth edition. Hoboken, NJ: Wiley (Wiley series in probability and statistics)."
},

{
    "location": "api/#",
    "page": "API References",
    "title": "API References",
    "category": "page",
    "text": ""
},

{
    "location": "api/#LsqFit.curve_fit",
    "page": "API References",
    "title": "LsqFit.curve_fit",
    "category": "function",
    "text": "curve_fit(model, xdata, ydata, p0) -> fit\n\nFit data to a non-linear model. p0 is an initial model parameter guess (see Example). The return object is a composite type (LsqFitResult), with some interesting values:\n\nfit.dof : degrees of freedom\nfit.param : best fit parameters\nfit.resid : residuals = vector of residuals\nfit.jacobian : estimated Jacobian at solution\n\nExample\n\n# a two-parameter exponential model\n# x: array of independent variables\n# p: array of model parameters\nmodel(x, p) = p[1]*exp.(-x.*p[2])\n\n# some example data\n# xdata: independent variables\n# ydata: dependent variable\nxdata = linspace(0,10,20)\nydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))\np0 = [0.5, 0.5]\n\nfit = curve_fit(model, xdata, ydata, p0)\n\n\n\n"
},

{
    "location": "api/#API-1",
    "page": "API References",
    "title": "API",
    "category": "section",
    "text": "Modules = [LsqFit]\nPrivate = false\nPages   = [\"curve_fit.jl\"]\nOrder   = [:function]"
},

]}
