#
# Correctness Tests
#

using CurveFit

my_tests = [ "levenberg_marquardt.jl",
             "curve_fit.jl"]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end
