using QuantumOptics, DifferentialEquations, LinearAlgebra, QuantumOpticsBase, LaTeXStrings, Plots, Statistics
include("quantum_parameter_estimation_lib.jl")

PLOTS_DIR = "/Users/henryhunt/Desktop/LabNotebooks/DickeModel/parameter_estimation/plots"

# Define parameters
function hideo_run_single(; ω=0.0, κ=2π * 30.0, g=2π * 45.0, ε=2π * 44.3, γperp=2π * 2.5)
    # Define basis and operators
    N = 10 # Truncation of Fock space
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

    H_nh = Heff
    for c_op in c_ops
        H_nh += -im / 2 * dagger(c_op) * c_op
    end

    # Initial state (coherent state)
    α = 0.0#im * ε / (κ / 2)
    ψ0 = tensor(coherentstate(fb, α), spindown(sb))

    # Do evolution with random jumps
    tlist = 0:0.00001:1
    tout, psi_t, jump_t, jump_index = timeevolution.mcwf(tlist, ψ0, Heff, c_ops, display_jumps=true)

    return tlist, ψ0, Heff, c_ops, tout, psi_t, jump_t, jump_index
end

function save_fig_1a()
    N = 10 # Truncation of Fock space
    fb = FockBasis(N)
    sb = SpinBasis(1 // 2)
    bases = [fb, sb]
    full_basis = tensor(fb, sb)
    a = mb(destroy(fb), bases, 1)
    σm = mb(sigmam(sb), bases, 2)
    idOp = mb(identityoperator(sb), bases, 2)
    ε_values = [2π * 24.0, 2π * 34.0, 2π * 44.3]
    fig = plot(layout=(3, 1), size=(800, 600))
    ylims!(fig, 0.0, 3.0)
    for (i, ε) in enumerate(ε_values)
        tlist, ψ0, Heff, c_ops, tout, psi_t, jump_t, jump_index = hideo_run_single(ε=ε)
        plot!(fig[i], tlist, real(expect(dagger(a) * a, psi_t)), xlabel="Time", ylabel=L"\langle a^\dag a\rangle", label="ε = 2π $(round(ε/2π,digits=2))")
    end

    display(fig)
    savefig(fig, joinpath(PLOTS_DIR, "hideo_fig1a.svg"))
    savefig(fig, joinpath(PLOTS_DIR, "hideo_fig1a.png"))
end

save_fig_1a()


function hideo_param_est(; ω=0.0, κ=2π * 30.0, gacc=2π * 45.0, glist=2π * LinRange(35, 57, 30), ε=2π * 44.3, γperp=2π * 2.5, tlist=0:0.0001:1)
    # Define basis and operators
    N = 10 # Truncation of Fock space
    fb = FockBasis(N)
    sb = SpinBasis(1 // 2)
    bases = [fb, sb]
    full_basis = tensor(fb, sb)

    a = mb(destroy(fb), bases, 1)
    σm = mb(sigmam(sb), bases, 2)
    idOp = mb(identityoperator(sb), bases, 2)

    # Define Hamiltonian
    Heff = im * gacc * (a * dagger(σm) - dagger(a) * σm) + im * ε * (a - dagger(a)) #- im  * γperp * dagger(σm) * σm - im  * κ * dagger(a) * a

    # Define collapse operators
    c_ops = [sqrt(2 * γperp) * σm, sqrt(2 * κ) * a]

    H_nh = Heff
    for c_op in c_ops
        H_nh += -im / 2 * dagger(c_op) * c_op
    end

    # Initial state (coherent state)
    α = 0.0#im * ε / (κ / 2)
    ψ0 = tensor(coherentstate(fb, α), spindown(sb))

    # Do evolution with random jumps
    tout, psi_t, jump_t, jump_index = timeevolution.mcwf(tlist, ψ0, Heff, c_ops, display_jumps=true)

    cumm_log_probs = zeros((length(tlist), length(glist)))

    for (j, g) in enumerate(glist)
        Heff = im * g * (a * dagger(σm) - dagger(a) * σm) + im * ε * (a - dagger(a))
        H_nh = Heff
        for c_op in c_ops
            H_nh += -im / 2 * dagger(c_op) * c_op
        end

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
        cumm_log_probs[i, :] .-= maximum(cumm_log_probs[i, :])
        max_likelihood_indices[i] = argmax(cumm_log_probs[i, :])
    end
    return cumm_log_probs, max_likelihood_indices
end

function save_fig1b()
    glist = 2π * LinRange(35, 57, 30)
    tlist = 0:0.0001:1
    gacc = 2π * 45.0
    cumm_log_probs, max_likelihood_indices = hideo_param_est(glist=glist, tlist=tlist, gacc=gacc)
    # Plot max likelihood g's over time
    fig = plot(layout=(2, 1), size=(800, 800), height=[0.2, 0.8])
    plot!(fig[1], tlist, map(i -> glist[i], max_likelihood_indices) ./ (2π), xlabel="Time (μs)", ylabel="g (MHz)", label="Max Likelihood g")
    ylims!(fig[1], glist[begin] / 2π, glist[end] / 2π)
    hline!(fig[1], [gacc / (2π)], linestyle=:dash, label="Ground Truth")
    # 3D plot of the probability of the different g's over time
    prob_matrix = exp.(cumm_log_probs)#map(prob_dist -> prob_dist ./ (sum(prob_dist) * (glist[2] - glist[1])), exp.(cumm_log_probs))
    surface!(fig[2], tlist, glist ./ (2π), prob_matrix', xlabel="Time (μs)", ylabel="g (MHz)", zlabel="Probability Density", label="Probability of g", xrotation=-60, colorbar=false, wireframe=true)
    zlims!(fig[2], 0.0, 1.0)
    display(fig)
    savefig(fig, joinpath(PLOTS_DIR, "hideo_fig1b.svg"))
    savefig(fig, joinpath(PLOTS_DIR, "hideo_fig1b.png"))
end

save_fig1b()

function save_fig2()
    glist = 2π * LinRange(35, 57, 30)
    tlist = 0:0.0001:1
    gacc = 2π * 45.0
    ε_values = [2π * 24.0, 2π * 34.0, 2π * 44.3]
    fig = plot(layout=(2, 1), size=(800, 800))
    ylabel!(fig[1], "Standard Deviation of MLE (MHz)")
    xlabel!(fig[1], "Time (μs)")
    start_t = 0.05
    num_runs = 100
    avg_prob_matrix = zeros(length(tlist), length(glist))
    for (i, ε) in enumerate(ε_values)
        max_likelihood_indices_runs_ε = zeros(Int, length(tlist), num_runs)
        avg_prob_matrix = zeros(length(tlist), length(glist))
        for run in 1:num_runs
            cumm_log_probs, max_likelihood_indices = hideo_param_est(glist=glist, tlist=tlist, gacc=gacc, ε=ε)
            max_likelihood_indices_runs_ε[:, run] = max_likelihood_indices
            avg_prob_matrix += exp.(cumm_log_probs)
        end

        std_dev_max_likelihood_ε = std(map(i -> glist[i], max_likelihood_indices_runs_ε), dims=2) ./ (2π)
        avg_prob_matrix ./= num_runs

        plot!(fig[1], tlist[tlist.>=start_t], std_dev_max_likelihood_ε[tlist.>=start_t], label="ε = 2π $(round(ε/2π,digits=2))")
    end
    # 3D plot of the average probability of the different g's over time
    surface!(fig[2], tlist[tlist.>=start_t], glist ./ (2π), avg_prob_matrix[tlist.>=start_t, :]', xlabel="Time (μs)", ylabel="g (MHz)", zlabel="Average Probability Density", label="Average Probability of g", colorbar=false, wireframe=true)
    zlims!(fig[2], 0.0, 1.0)
    display(fig)
    savefig(fig, joinpath(PLOTS_DIR, "hideo_fig2.svg"))
    savefig(fig, joinpath(PLOTS_DIR, "hideo_fig2.png"))
end

save_fig2()