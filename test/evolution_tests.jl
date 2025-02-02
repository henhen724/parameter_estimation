using TestSetExtensions, LinearAlgebra, QuantumOptics
include("../states/index.jl")
include("../evolution/index.jl")

# @testset "Evolution Tests" begin

# end


ω = 0.0;
κ = 2π * 30.0;
g = 2π * 45.0;
ε = 2π * 44.3;
γperp = 2π * 2.5;

# Define basis and operators
N = 3 # Truncation of Fock space
fb = FockBasis(N)
sb = SpinBasis(1 // 2)
bases = [fb, sb]
full_basis = tensor(fb, sb)

a = mb(destroy(fb), bases, 1)
σm = mb(sigmam(sb), bases, 2)
idOp = mb(identityoperator(sb), bases, 2)

# Define Hamiltonian
Heff = im * g * (a * dagger(σm) - dagger(a) * σm) + im * ε * (a - dagger(a)) #- im  * γperp * dagger(σm) * σm - im  * κ * dagger(a) * a

# Define collapse operators
c_ops = [sqrt(2 * γperp) * σm, sqrt(2 * κ) * a]

# Initial state (coherent state)
α = 0.0#im * ε / (κ / 2)
ψ0 = tensor(coherentstate(fb, α), spindown(sb))

# Do evolution with random jumps
tlist = 0:0.0001:1
tout, ρ_t = timeevolution.master(tlist, dm(ψ0), Heff, c_ops)

ρ0 = dm(ψ0)
H = Heff

HS_dim = size(ρ0, 1)

H_data = H.data
c_ops_data = map(x -> x.data, c_ops)
function lindblad_rhs!(dstate, state::MasterEquationState, p, t)
    ρ = state.rho
    dstate.rho .= -im * (H_data * ρ - ρ * H_data)
    for c_op in c_ops_data
        dstate.rho .+= c_op * ρ * c_op - 0.5 * (adjoint(c_op) * c_op * ρ + ρ * adjoint(c_op) * c_op)
    end
end

# Convert initial density matrix to vector form
state0 = MasterEquationState(ρ0.data, 0.0im)

# Define the ODE problem
prob = ODEProblem{true,MasterEquationState}(lindblad_rhs!, state0, (tlist[1], tlist[end]))

# Solve the ODE problem
sol = solve(prob, Tsit5(), saveat=tlist)

# Convert solution back to density matrix form
ρ_t = [Operator(basis(ρ0), reshape(sol.u[i], size(ρ0))) for i in 1:length(sol.u)]


tout_2, ρ_t2 = integrate_master_equation(tlist, dm(ψ0), Heff, c_ops)

ρ0 = dm(ψ0)
H_data = Heff.data
c_ops_data = map(x -> x.data, c_ops)
function lindblad_rhs(dstate, state::MasterEquationState, p, t)
    ρ = state.rho
    dstate.rho .= -im * (H_data * ρ - ρ * H_data)
    for c_op in c_ops_data
        dstate.rho .+= c_op * ρ * c_op - 0.5 * (adjoint(c_op) * c_op * ρ + ρ * adjoint(c_op) * c_op)
    end
end

# Convert initial density matrix to vector form
state0 = MasterEquationState(ρ0.data, 0.0im)

# Define the ODE problem
prob = ODEProblem(lindblad_rhs, state0, (tlist[1], tlist[end]))

# Solve the ODE problem
sol = solve(prob, saveat=tlist)
