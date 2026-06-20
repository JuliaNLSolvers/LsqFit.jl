#
# Correctness Tests
#
using LsqFit, Test, LinearAlgebra
using NLSolversBase

my_tests = [
    "curve_fit.jl",
    "weights.jl",
    "levenberg_marquardt.jl",
    "curve_fit_inplace.jl",
    "geodesic.jl",
    "qa.jl",
]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end
