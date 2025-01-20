using DifferentialEquations

struct MasterEqState <: AbstractArray{ComplexF64,2}
    HS_dim::Int
    ρ::Matrix{ComplexF64}
    logtrρ::ComplexF64
end

function MasterEqState(ρ::Matrix{ComplexF64})::MasterEqState
    return MasterEqState(size(ρ, 1), ρ, 0.0)
end

Base.size(var::MasterEqState)::NTuple{2,Int} = (var.HS_dim, var.HS_dim)

Base.IndexStyle(::Type{MasterEqState}) = IndexCartesian()
Base.getindex(var::MasterEqState, I::CartesianIndex)::ComplexF64 = var.ρ[I]
Base.setindex!(var::MasterEqState, v, I::CartesianIndex) = (var.ρ[I] = v)

Base.similar(var::MasterEqState) = MasterEqState(var.HS_dim, similar(var.ρ, ComplexF64), 0.0)
Base.similar(var::MasterEqState, ::Type{T}) where {T} = MasterEqState(var.HS_dim, similar(var.ρ, T), 0.0)
Base.getindex(var::MasterEqState, i::Int, j::Int)::ComplexF64 = var.ρ[i, j]

function my_dynamics(du, u, p, t)
    dρ .= 0.0
end

# Step 3: Define the initial state and time span

ρ0 = MasterEqState(ComplexF64[[1.0 0.0] [0.0 1.0]])
tspan = (0.0, 10.0)

# Step 4: Solve the differential equation
prob = ODEProblem(my_dynamics, ρ0, tspan)
sol = solve(prob, Tsit5())

# Print the solution
for (t, u) in zip(sol.t, sol.u)
    println("t = $t, x = $(u.x), y = $(u.y)")
end