using DifferentialEquations

struct MyState <: AbstractArray{ComplexF64,1}
    lengths::NTuple{2,Int}
    x::Vector{ComplexF64}
    y::Vector{ComplexF64}
end
function MyState(x::Vector{ComplexF64}, y::Vector{ComplexF64})::MyState
    return MyState((length(x), length(y)), x, y)
end

Base.size(foo::MyState)::Tuple{Int} = (foo.lengths[1] + foo.lengths[2],)
Base.IndexStyle(::Type{<:MyState}) = IndexLinear()
function Base.getindex(foo::MyState, i::Int)::ComplexF64
    if i <= length(foo.x)
        foo.x[i]
    else
        foo.y[i-foo.lengths[1]]
    end
end

function Base.setindex!(var::MyState, v, i::Int)
    if i <= length(var.x)
        var.x[i] = v
    else
        var.y[i-var.lengths[1]] = v
    end
end

Base.similar(foo::MyState) = MyState(foo.lengths, similar(foo.x), similar(foo.y))
Base.similar(foo::MyState, ::Type{T}) where {T} = MyState(foo.lengths, similar(foo.x, T), similar(foo.y, T))

# Step 2: Define the differential equation function
function my_dynamics(du, u, p, t)
    du.x .= u.y
    du.y .= -u.x
end

# Step 3: Define the initial state and time span
u0 = MyState(zeros(ComplexF64, 3), zeros(ComplexF64, 3))
tspan = (0.0, 10.0)

# Step 4: Solve the differential equation
prob = ODEProblem(my_dynamics, u0, tspan)
sol = solve(prob, Tsit5())

# Print the solution
for (t, u) in zip(sol.t, sol.u)
    println("t = $t, x = $(u.x), y = $(u.y)")
end