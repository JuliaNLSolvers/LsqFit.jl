#
# Correctness Tests
#
using LsqFit, Test, LinearAlgebra
using NLSolversBase

my_tests = ["curve_fit.jl", "levenberg_marquardt.jl", "curve_fit_inplace.jl", "geodesic.jl", "measurements.jl"]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end
