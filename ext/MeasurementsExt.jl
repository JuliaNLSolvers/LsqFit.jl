module MeasurementsExt

using LsqFit
isdefined(Base, :get_extension) ? (using Measurements) : (using ..Measurements)

function LsqFit.curve_fit(
    model,
    xdata::AbstractArray,
    ydata::AbstractArray{Measurement{T}} where T,
    p0::AbstractArray;
    inplace=false,
    kwargs...,
)
    y = Measurements.value.(ydata)
    ye = Measurements.uncertainty.(ydata)
    wt = ye .^ -2
    curve_fit(model, xdata, y, wt, p0; inplace=inplace, kwargs...)
end

function LsqFit.curve_fit(
    model,
    jacobian,
    xdata::AbstractArray,
    ydata::AbstractArray{Measurement{T}} where T,
    p0::AbstractArray;
    inplace=false,
    kwargs...,
)
    y = Measurements.value.(ydata)
    ye = Measurements.uncertainty.(ydata)
    wt = ye .^ -2
    curve_fit(model, jacobian, xdata, y, wt, p0; inplace=inplace, kwargs...)
end

end # module