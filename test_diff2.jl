using DifferentialEquations

struct my_type{A,N} <: AbstractArray{A,N}
    data::Vector{A}
    name::String
end
my_type(data::AbstractArray{T,N}, name::String) where {T,N} =
    my_type{eltype(data),N}(data, name)

Base.size(var::my_type) = size(var.data)

Base.getindex(var::my_type, i::Int) = var.data[i]
# Base.getindex(var::my_type, I::Vararg{Int,N}) where {N} = var.data[I...]
# Base.getindex(var::my_type, ::Colon) = var.data[:]
# Base.getindex(var::my_type, kr::AbstractRange) = var.data[kr]

Base.setindex!(var::my_type, v, i::Int) = (var.data[i] = v)
# Base.setindex!(var::my_type, v, I::Vararg{Int,N}) where {N} = (var.data[I...] = v)
# Base.setindex!(var::my_type, v, ::Colon) = (var.data[:] .= v)
# Base.setindex!(var::my_type, v, kr::AbstractRange) = (var.data[kr] .= v)

function rhs_test(f, p, t)
    f
end

xmin = -2.0 * pi
xmax = 2.0 * pi
xnodes = 600
hx = (xmax - xmin) / xnodes

xx = range(xmin, stop=xmax, length=600)

x0 = 0
w = 0.4
A = 1

f0 = A * exp.(-((xx .- x0) ./ w) .^ 2)
foo = my_type(f0, "foo")

tspan = (0.0, 1.0)

Base.similar(foo::my_type) = my_type(similar(foo.data), foo.name)
Base.similar(foo::my_type, ::Type{T}) where {T} = my_type(similar(foo.data, T), foo.name)
function rhs_test2(df, f, p, t)
    println(typeof(f))
    df.data .= f.data
end
prob = ODEProblem(rhs_test2, foo, tspan)
sol = solve(prob, RK4())