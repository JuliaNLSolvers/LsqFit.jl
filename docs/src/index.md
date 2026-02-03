# LsqFit.jl

Basic least-squares fitting in pure Julia under an MIT license.

The basic functionality was originaly in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), before being separated into this library.  At this time, `LsqFit` only utilizes the Levenberg-Marquardt algorithm for non-linear fitting.

`LsqFit.jl` is part of the [JuliaNLSolvers](https://github.com/JuliaNLSolvers) family.

|Source|Package Evaluator|Build Status|
|:----:|:---------------:|:----------:|
| [![Source](https://img.shields.io/badge/GitHub-source-green.svg)](https://github.com/JuliaNLSolvers/Optim.jl) |[![LsqFit](http://pkg.julialang.org/badges/LsqFit_0.4.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.4) [![LsqFit](http://pkg.julialang.org/badges/LsqFit_0.5.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.5) [![LsqFit](http://pkg.julialang.org/badges/LsqFit_0.6.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.6) [![LsqFit](http://pkg.julialang.org/badges/LsqFit_0.7.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.7)|[![Build Status](https://travis-ci.org/JuliaNLSolvers/LsqFit.jl.svg)](https://travis-ci.org/JuliaNLSolvers/LsqFit.jl)|

## Install

To install the package, run

```julia
Pkg.add("LsqFit")
```

If you want the latest features, also run

```julia
Pkg.checkout("LsqFit")
```

To use the package in your code

```julia-repl
julia> using LsqFit
```
