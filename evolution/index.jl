include("../states/index.jl")
import QuantumOpticsBase: Operator
using DifferentialEquations

"""
    integrate_master_equation(H, c_ops, ρ0, tlist)

Integrates the master equation for a given Hamiltonian and collapse operators over a specified time list.

# Arguments
- `H::Operator`: The Hamiltonian of the system.
- `c_ops::Vector{Operator}`: A vector of collapse operators.
- `ρ0::Operator`: The initial density matrix of the system.
- `tlist::Vector{Float64}`: A list of time points at which to evaluate the solution.

# Returns
- `tlist::Vector{Float64}`: The list of time points.
- `ρ_t::Vector{Operator}`: The density matrices at each time point in `tlist`.

# Example
"""
function integrate_master_equation(tlist, ρ0::Operator, H::Operator, c_ops::Vector{<:Operator})
    # Define the Lindblad superoperator
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
    prob = ODEProblem{MasterEquationState}(lindblad_rhs!, state0, (tlist[1], tlist[end]))

    # Solve the ODE problem
    sol = solve(prob, Tsit5(), saveat=tlist)

    # Convert solution back to density matrix form
    ρ_t = [Operator(basis(ρ0), reshape(sol.u[i], size(ρ0))) for i in 1:length(sol.u)]

    return tlist, ρ_t
end