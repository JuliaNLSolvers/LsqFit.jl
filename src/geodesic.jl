# see related https://discourse.julialang.org/t/nested-forwarddiff-jacobian-calls-with-inplace-function/21232/7
function make_out_of_place_func(f!, x, make_buffer)
    y = make_buffer(f!, x) # make_buffer overloads will be defined below
    x -> f!(y, x)
end

struct OutOfPlace{F,D}
    f!::F
    out_of_place_funcs::Dict{Type,D}
    make_buffer::D

    function OutOfPlace(f!::T, dict::Dict{Type,D}, shape::Tuple) where {T,D}

        mk(::T, p) = similar(p, shape)
        new{T,D}(f!, dict, mk)
    end
end

OutOfPlace(f!, shape) = OutOfPlace(f!, Dict{Type,Function}(), shape) #no sure how to remove the `Function` here

eval_f(f::F, x) where {F} = f(x) # function barrier
function (oop::OutOfPlace{F})(x) where {F}
    T = eltype(x)
    f = get!(
        () -> make_out_of_place_func(oop.f!, x, oop.make_buffer),
        oop.out_of_place_funcs,
        T,
    )
    eval_f(f, x)
end



function make_hessian(f!, x0, p0)

    f = OutOfPlace(f!, (length(x0),))
    g! = (outjac, p) -> (ForwardDiff.jacobian!(outjac, f, p); outjac = outjac')
    g = OutOfPlace(g!, (length(x0), length(p0)))
    h! = (outjac2, p) -> (ForwardDiff.jacobian!(outjac2, g, p);
    reshape(outjac2', length(p0), length(p0), length(x0)))
    h!
end

struct Avv
    h!::Function
    hessians::Array{Float64}

    function Avv(h!::Function, n::Int, m::Int)
        hessians = Array{Float64}(undef, m * n, n)
        new(h!, hessians)
    end
end

function (avv::Avv)(dir_deriv::AbstractVector, p::AbstractVector, v::AbstractVector)
    hess = avv.h!(avv.hessians, p) #half of the runtime
    vHv!(dir_deriv, hess, v) #half of the runtime, almost all the memory
end

function vHv!(dir_deriv::AbstractVector, hessians::AbstractArray, v::AbstractVector)
    tmp = similar(v) #v shouldn't be too large in general, so I kept it here
    vt = v'

    for i = 1:length(dir_deriv)
        @views mul!(tmp, hessians[:, :, i], v) #this line is particularly expensive memory-wise
        dir_deriv[i] = vt * tmp
    end

end
