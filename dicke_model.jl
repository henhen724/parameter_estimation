using QuantumOptics, DifferentialEquations, LinearAlgebra, QuantumOpticsBase, LaTeXStrings, Plots, Statistics
include("quantum_parameter_estimation_lib.jl")

PLOTS_DIR = "/Users/henryhunt/Desktop/LabNotebooks/DickeModel/parameter_estimation/plots"

function make_operators(fockmax, Nspin)
    fb = FockBasis(fockmax)
    sb = SpinBasis(Nspin // 2)
    bases = [sb, fb]
    a = mb(destroy(fb), bases, 2)
    Sx = mb(sigmax(sb), bases, 1) / 2
    Sy = mb(sigmay(sb), bases, 1) / 2
    Sz = mb(sigmaz(sb), bases, 1) / 2
    idOp = mb(identityoperator(sb), bases, 1)
    return fb, sb, bases, a, Sx, Sy, Sz, idOp
end

# Define basis and operators
fockmax = 10 # Truncation of Fock space
Nspin = 10
fb, sb, bases, a, Sx, Sy, Sz, idOp = make_operators(fockmax, Nspin)
σm = mb(sigmam(sb), bases, 1)
full_basis = tensor(sb, fb)

κ = 2π * 0.15 #2π * 0.15
Δc = -2π * 20
ωz = 2π * 0.01
ε = 2π * 44.3
γperp = 2π * 2.5

#gc = sqrt((Δc^2 + κ^2) / abs(Δc) * ωz)
glist = 2π * LinRange(37, 57, 30)#gc * LinRange(0.5, 1.5, 30)
gacc = 2π * 45#glist[24]

# Define Hamiltonian
H = Δc * dagger(a) * a + ωz * Sz + ε * (a + dagger(a)) + (gacc * dagger(a) + gacc * a) * Sx / sqrt(Nspin)

# Define collapse operators
c_ops = [sqrt(2 * γperp) * σm, sqrt(2 * κ) * a]

# H_nh = H
# for c_op in c_ops
#     H_nh += -im / 2 * dagger(c_op) * c_op
# end

# Initial state (coherent state)
α = 0.0#im * ε / (κ / 2)
ψ0 = tensor(spindown(sb), coherentstate(fb, α))

# Do evolution with random jumps
tlist = 0:0.0001:10.0
tout, psi_t, jump_t, jump_index = timeevolution.mcwf(tlist, ψ0, H, c_ops, display_jumps=true)

# Plot the average photon number
plot(tout[begin:2:end], real.(expect(dagger(a) * a, psi_t))[begin:2:end], xlabel="Time", ylabel="Average Photon Number", title="Average Photon Number vs Time")

cumm_log_probs = zeros((length(tlist), length(glist)))

for (j, g) in enumerate(glist)
    H = Δc * dagger(a) * a + ωz * Sz + ε * (a + dagger(a)) + (g * dagger(a) + g * a) * Sx / sqrt(Nspin)
    H_nh = H
    for c_op in c_ops
        H_nh += -im / 2 * dagger(c_op) * c_op
    end
    println("Runing g $(g)")
    sol, jump_probs = integrate_schrodinger_with_record(H_nh, c_ops, tlist, ψ0, jump_t, jump_index)
    # Convert solution back to QuantumOptics states
    psi_t2 = [Ket(full_basis, sol.u[i]) for i in 1:length(sol.u)]
    log_prob_from_jumps = cumsum(log.(jump_probs))
    log_norm = log.(expect(idOp, psi_t2))
    for (t_indx, t) in enumerate(tlist)
        last_jump = findlast(j_t -> j_t < t, jump_t)
        if last_jump isa Nothing
            cumm_log_probs[t_indx, j] = log_norm[t_indx]
        else
            cumm_log_probs[t_indx, j] = log_norm[t_indx] + log_prob_from_jumps[last_jump]
        end
    end
end

max_likelihood_indices = zeros(Int, length(tlist))
for i in 1:length(tlist)
    # cumm_log_probs[i, :] .-= maximum(cumm_log_probs[i, :])
    max_likelihood_indices[i] = argmax(cumm_log_probs[i, :])
end

# gc = 2π

fig = plot(layout=(2, 1), size=(800, 800), height=[0.2, 0.8])
plot!(fig[1], tlist, map(i -> glist[i], max_likelihood_indices) ./ gc, xlabel="Time (μs)", ylabel="\$g/g_c\$", label="Max Likelihood g")
ylims!(fig[1], glist[begin] / gc, glist[end] / gc)
hline!(fig[1], [gacc / gc], linestyle=:dash, label="Ground Truth")
# 3D plot of the probability of the different g's over time
prob_matrix = exp.(cumm_log_probs)#map(prob_dist -> prob_dist ./ (sum(prob_dist) * (glist[2] - glist[1])), exp.(cumm_log_probs))
surface!(fig[2], tlist, glist ./ gc, prob_matrix', xlabel="Time (μs)", ylabel="\$g/g_c\$", zlabel="Probability Density", label="Probability of g", xrotation=-60, colorbar=false, wireframe=true)
zlims!(fig[2], 0.0, 1.0)
display(fig)
savefig(fig, joinpath(PLOTS_DIR, "WithEpsilon$(round(ε, digits=2))ParamEstimation.png"))