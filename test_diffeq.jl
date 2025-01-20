using DifferentialEquations

struct MyState <: AbstractArray{ComplexF64,1}
    lengths::NTuple{2,Int}
    x::Matrix{ComplexF64}
    y::Vector{ComplexF64}
end
# function MyState(x::Vector{ComplexF64}, y::Vector{ComplexF64})::MyState
#     return MyState((length(x), length(y)), x, y)
# end

Base.size(foo::MyState) = size(foo.x)
Base.IndexStyle(::Type{<:MyState}) = IndexCartesian()
function Base.getindex(foo::MyState, i::Int, j::Int)::ComplexF64
    return foo.x[i, j]
end
function Base.getindex(foo::MyState, I::CartesianIndex)::ComplexF64
    return foo.x[I]
end

function Base.setindex!(var::MyState, v, i::Int, j::Int)
    var.x[i, j] = v
    return
end

Base.similar(foo::MyState) = MyState(foo.lengths, similar(foo.x), similar(foo.y))
Base.similar(foo::MyState, ::Type{T}) where {T} = MyState(foo.lengths, similar(foo.x, T), similar(foo.y, T))

# Step 2: Define the differential equation function
function my_dynamics(du, u::MyState, p, t)
    du.x .= u.y
    du.y .= -u.x
end

# Step 3: Define the initial state and time span
u0 = MyState((3, 3), zeros(ComplexF64, (3, 3)), zeros(ComplexF64, 3))
tspan = (0.0, 10.0)

# Step 4: Solve the differential equation
prob = ODEProblem(my_dynamics, u0, tspan)
sol = solve(prob, Tsit5())

# Print the solution
for (t, u) in zip(sol.t, sol.u)
    println("t = $t, x = $(u.x), y = $(u.y)")
end