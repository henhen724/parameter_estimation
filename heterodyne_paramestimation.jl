using QuantumOptics, DifferentialEquations, LinearAlgebra, QuantumOpticsBase, LaTeXStrings, Plots, Statistics
include("quantum_parameter_estimation_lib.jl")

N = 10 # Truncation of Fock space
fb = FockBasis(N)
sb = SpinBasis(1 // 2)
bases = [fb, sb]
full_basis = tensor(fb, sb)

a = mb(destroy(fb), bases, 1)
σm = mb(sigmam(sb), bases, 2)
idOp = mb(identityoperator(sb), bases, 2)

ω = 0.0
κ = 2π * 30.0
g = 2π * 45.0
ε = 2π * 44.3
γperp = 2π * 2.5

# Define Hamiltonian
H = im * g * (a * dagger(σm) - dagger(a) * σm) + im * ε * (a - dagger(a))

# Define collapse operators
c_ops = [sqrt(2 * γperp) * σm, sqrt(2 * κ) * a]

# Initial state (coherent state)
α = 0.0#im * ε / (κ / 2)
ψ0 = tensor(coherentstate(fb, α), spindown(sb))

# Do evolution with random jumps
tlist = 0:0.0001:1.0
# dnorm = zeros(ComplexF64, length(tlist))

c_op_mats = [c_op.data for c_op in c_ops]
# Define the SDE problem
schrodinger_rhs! = let Hmat = H.data, c_op_mats = c_op_mats, idMat = idOp.data
    function schrodinger_rhs!(du, u, p, t)
        H_nl = Hmat
        for (idx, c_op_mat) in enumerate(c_op_mats)
            C_expect = (conj.(u)' * (c_op_mat) * u) / (conj.(u)' * u)
            # mean subtracted op
            c_op_mean_sub = c_op_mat - C_expect * idMat
            H_nl += im * (C_expect + conj(C_expect)) * c_op_mean_sub
            # println("norm $(conj.(u)' * u)\t", " expect of C - C_mean ", conj.(u)' * (c_op_mean_sub.data) * u)
            H_nl += -im / 2 * (conj.(c_op_mean_sub)' * c_op_mean_sub)
        end
        mul!(du, H_nl, u, -1.0im, 0.0im)
        # println(-im * H_nl)
        # println("du: $(du) \t u: $(u)")
        # println("norm of u $(conj.(u)' * u)")
        # println("deterministic change in norm ", (conj.(u)' * du + conj.(du)' * u) / (conj.(u)' * u))
    end
end

noise_func! = let c_op_mats = c_op_mats
    function noise_func!(du, u, p, t)
        # du .= 0.0
        for (idx, c_op_mat) in enumerate(c_op_mats)
            # mul!(du[:, idx], c_op.data, u, 1.0, 0.0)
            du[:, idx] = c_op_mat * u
            C_expect = (conj.(u)' * c_op_mat * u) / (conj.(u)' * u)
            du[:, idx] -= C_expect * u
            # println("stoch. change in norm ", (conj.(u)' * du[:, idx] + conj.(du[:, idx])' * u) / (conj.(u)' * u))
            # println("ito change in norm ", conj.(du[:, idx])' * du[:, idx] / (conj.(u)' * u))
        end
    end
end

function norm_func(u, t, integrator)
    # println("Normalizing u")
    integrator.u = u / sqrt(conj.(u)' * u)
end

ncb = FunctionCallingCallback(norm_func;
    func_everystep=true,
    func_start=false)
prob = SDEProblem(schrodinger_rhs!, noise_func!, ψ0.data, (0.0, 1.0), noise_rate_prototype=zeros((size(ψ0)..., length(c_ops))))

# Solve the SDE problem
sol = solve(prob, RKMilGeneral(; ii_approx=IICommutative()); saveat=tlist, dt=2^-15, adaptive=false, callback=ncb)

# Plot the results
psi_t = [Ket(full_basis, sol.u[i]) for i in 1:length(sol.u)]
normilization = real.(expect(idOp, psi_t))
plot(normilization, label=L"\langle 1 \rangle")
plot(real.(expect(dagger(a) * a, psi_t)), label=L"\langle a^\dagger a \rangle")
plot(real.(expect(dagger(σm) * σm, psi_t)), label=L"\langle \sigma_+ \sigma_- \rangle")
plot(real.(expect(dagger(c_ops[1]) + c_ops[1], psi_t)))
plot(real.(expect(dagger(c_ops[2]) + c_ops[2], psi_t)))

# Plot the Fock state probabilities
fock_probs = [real(diag(ptrace(dm(psi_t[i] / sqrt(normilization[i])), 2).data)) for i in 1:length(psi_t)]
plot(tlist, hcat(fock_probs...)', label=hcat(["P(n=$i)" for i in 0:N]...))