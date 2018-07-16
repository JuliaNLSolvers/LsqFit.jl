#
# Correctness Tests
#

using LsqFit, Test, LinearAlgebra, Random

my_tests = [ "curve_fit.jl", "levenberg_marquardt.jl"]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end
