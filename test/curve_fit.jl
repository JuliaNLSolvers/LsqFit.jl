# fitting noisy data to an exponential model
model(x, p) = p[1]*exp(-x.*p[2])

# some example data
srand(12345)
xdata = linspace(0,10,20)
ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))

fit = curve_fit(model, xdata, ydata, [0.5, 0.5])
@assert norm(fit.param - [1.0, 2.0]) < 0.05

# can also get error estimates on the fit parameters
errors = estimate_errors(fit)
@assert norm(errors - [0.017, 0.075]) < 0.01
