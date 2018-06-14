# LsqFit.jl

Basic least-squares fitting in pure Julia under an MIT license.

The basic functionality was originaly in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), before being separated into this library.  At this time, `LsqFit` only utilizes the Levenberg-Marquardt algorithm for non-linear fitting.

`LsqFit.jl` is part of the [JuliaNLSolvers](https://github.com/JuliaNLSolvers) family.

|Package Evaluator|Build Status|
|:---------------:|:----------:|
|[![LsqFit](http://pkg.julialang.org/badges/LsqFit_0.6.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.6) [![LsqFit](http://pkg.julialang.org/badges/LsqFit_0.7.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.7)|[![Build Status](https://travis-ci.org/JuliaNLSolvers/LsqFit.jl.svg)](https://travis-ci.org/JuliaNLSolvers/LsqFit.jl)|

## Install

To install the package, run

```julia
Pkg.add("LsqFit")
```

If you want the latest features, also run

```julia
Pkg.checkout("LsqFit")
```
