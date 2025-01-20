using QuantumOptics, DifferentialEquations, LinearAlgebra, QuantumOpticsBase, LaTeXStrings, Plots, Statistics
include("quantum_parameter_estimation_lib.jl")

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

H_nh = Heff
for c_op in c_ops
    H_nh += -im / 2 * dagger(c_op) * c_op
end

# Initial state (coherent state)
α = 0.0#im * ε / (κ / 2)
ψ0 = tensor(coherentstate(fb, α), spindown(sb))

# Do evolution with random jumps
tlist = 0:0.0001:1
tout, psi_t, jump_t, jump_index = timeevolution.mcwf(tlist, ψ0, Heff, c_ops, display_jumps=true)

function integrate_master_equation(H, c_ops, ρ0, tlist)
    # Define the Lindblad superoperator
    HS_dim = size(ρ0, 1)
    H_data = H.data
    c_ops_data = map(x -> x.data, c_ops)
    function lindblad_rhs!(dρ_vec, ρ_vec, p, t)
        ρ = reshape(ρ_vec, (HS_dim, HS_dim))
        dρ = similar(ρ)
        dρ .= -im * (H_data * ρ - ρ * H_data)
        for c_op in c_ops_data
            dρ .+= c_op * ρ * c_op - 0.5 * (adjoint(c_op) * c_op * ρ + ρ * adjoint(c_op) * c_op)
        end
        dρ_vec .= vec(dρ)
    end

    # Convert initial density matrix to vector form
    ρ0_vec = vec(ρ0.data)

    # Define the ODE problem
    prob = ODEProblem(lindblad_rhs!, ρ0_vec, (tlist[1], tlist[end]))

    # Solve the ODE problem
    sol = solve(prob, Tsit5(), saveat=tlist)

    # Convert solution back to density matrix form
    ρ_t = [Operator(basis(ρ0), reshape(sol.u[i], size(ρ0))) for i in 1:length(sol.u)]

    return tlist, ρ_t
end

struct MasterEqState
    ρ::Matrix{ComplexF64}
    logtrρ::Float64
end

function integrate_photon_counting_master_equation_with_record(H, c_ops, ρ0, tlist, jump_t, jump_index)
    # Define the Lindblad superoperator
    HS_dim = size(ρ0, 1)
    # println("The Hilbert space dimension is ", HS_dim)
    function to_state!(state, trρ, ρ_vec)
        state[1] = trρ
        state[2:HS_dim^2+1] .= ρ_vec
    end
    function to_state_alloc!(trρ, ρ_vec)
        state = zeros(ComplexF64, 1 + HS_dim^2)
        state[1] = trρ
        state[2:HS_dim^2+1] .= ρ_vec
        return state
    end
    function from_state!(state)
        return state[1], state[2:HS_dim^2+1]
    end
    function ρ_from_state!(state)
        return reshape(state[2:HS_dim^2+1], size(ρ0))
    end
    H_data = H.data
    c_ops_data = map(x -> x.data, c_ops)
    function lindblad_rhs!(dstate, state, p, t)
        logtrρ, ρ_vec = from_state!(state)
        ρ = reshape(ρ_vec, (HS_dim, HS_dim))
        dlogtrρ = 0.0im
        dρ = similar(ρ)
        dρ .= -im * (H_data * ρ - ρ * H_data)
        for c_op in c_ops_data
            leftcop = adjoint(c_op) * c_op * ρ
            norm_change = tr(leftcop)
            dρ .+= norm_change * ρ - 0.5 * (leftcop + ρ * adjoint(c_op) * c_op)
            dlogtrρ = -norm_change / tr(ρ)
        end
        to_state!(dstate, dlogtrρ, vec(dρ))
    end

    jump_probs = zeros(length(jump_t))
    tmp_x = similar(ρ0.data)
    function dojump!(integrator)
        x = integrator.u
        t = integrator.t
        time_index = findmin(j_t -> abs(j_t - t), jump_t)[2]
        logtrρ, ρ_vec = from_state!(x)
        ρ = reshape(ρ_vec, (HS_dim, HS_dim))
        tmp_x = c_ops[jump_index[time_index]].data * ρ * dagger(c_ops[jump_index[time_index]]).data
        # print(tr(tmp_x))
        tmp_norm = real.(tr(tmp_x))
        if tmp_norm > 10^2 * eps(eltype(tmp_norm))
            jump_probs[time_index] = tmp_norm * exp(logtrρ)
            to_state!(integrator.u, 0.0, vec(tmp_x ./ tmp_norm))
        else
            jump_probs[time_index] = 0.0
            to_state!(integrator.u, 0.0, vec(tmp_x ./ tmp_norm))
        end
    end

    jump_cb = PresetTimeCallback(jump_t, dojump!, save_positions=(false, false))

    function norm_func(u, t, integrator)
        logtrρ, ρ_vec = from_state!(u)
        ρ = reshape(ρ_vec, (HS_dim, HS_dim))
        to_state!(integrator.u, logtrρ, vec(ρ ./ tr(ρ)))
        return integrator.u
    end

    ncb = FunctionCallingCallback(norm_func; func_everystep=true, func_start=false)
    full_cb = CallbackSet(jump_cb, ncb)

    # Convert initial density matrix to vector form
    ρ0_vec = vec(ρ0.data)

    # Define the ODE problem
    prob = ODEProblem(lindblad_rhs!, to_state_alloc!(0.0, ρ0_vec), (tlist[1], tlist[end]), saveat=tlist)

    # Solve the ODE problem
    sol = solve(prob, Tsit5(); reltol=1e-5, abstol=1e-5, saveat=tlist, save_everystep=false, callback=full_cb, tstops=jump_t)

    println("Solution length: ", length(sol))

    # Convert solution back to density matrix form
    ρ_t = [Operator(basis(ρ0), ρ_from_state!(sol.u[i])) for i in 1:length(sol.u)]
    logtrρ_t = [sol.u[i][1] for i in 1:length(sol.u)]

    return tlist, ρ_t, logtrρ_t, jump_probs
end

function master_param_est(; ω=0.0, κ=2π * 30.0, gacc=2π * 45.0, glist=2π * LinRange(35, 57, 30), ε=2π * 44.3, γperp=2π * 2.5, tlist=0:0.0001:1, N=10)
    # Define basis and operators
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

    # Initial state (coherent state)
    α = 0.0#im * ε / (κ / 2)
    ψ0 = tensor(coherentstate(fb, α), spindown(sb))

    ρ0 = dm(ψ0)

    # Do evolution with random jumps
    tout, psi_t, jump_t, jump_index = timeevolution.mcwf(tlist, ψ0, Heff, c_ops, display_jumps=true)

    cumm_log_probs = zeros((length(tlist), length(glist)))

    for (j, g) in enumerate(glist)
        Heff = im * g * (a * dagger(σm) - dagger(a) * σm) + im * ε * (a - dagger(a))

        tlist, ρ_t, logtrρ_t, jump_probs = integrate_photon_counting_master_equation_with_record(Heff, c_ops, ρ0, tlist, jump_t, jump_index)
        log_prob_from_jumps = cumsum(log.(jump_probs))
        if length(tlist) > length(logtrρ_t)
            finish_logtrρ_t = repeat(Float64[-Inf], length(tlist) - length(logtrρ_t))
            logtrρ_t = vcat([logtrρ_t..., finish_logtrρ_t...])
        end
        for (t_indx, t) in enumerate(tlist)
            last_jump = findlast(j_t -> j_t < t, jump_t)
            if last_jump isa Nothing
                cumm_log_probs[t_indx, j] = logtrρ_t[t_indx]
            else
                cumm_log_probs[t_indx, j] = logtrρ_t[t_indx] + log_prob_from_jumps[last_jump]
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

# Define initial density matrix (pure state)
ρ0 = dm(ψ0)

# Integrate master equation
tlist, ρ_t, logtrρ_t, jump_probs_master = integrate_photon_counting_master_equation_with_record(Heff, c_ops, ρ0, tlist, jump_t, jump_index)

plot(tlist, real.(expect(idOp, ρ_t)), label="Stochastic Master Equation with Record", xlabel="Time", ylabel=L"\langle 1 \rangle")

p = plot(tlist, real.(expect(dagger(a) * a, ρ_t) ./ expect(idOp, ρ_t)), label="Stochastic Master Equation with Record", xlabel="Time", ylabel="Photon Number")
plot!(p, tlist, real.(expect(dagger(a) * a, psi_t)), label="Stochastic Schodenger Equation")

sol, jump_probs_sch = integrate_schrodinger_with_record(H_nh, c_ops, tlist, ψ0, jump_t, jump_index)
psi_t2 = [Ket(full_basis, sol.u[i]) for i in 1:length(sol.u)]
p = plot(tlist, real.(expect(dagger(a) * a, psi_t2) ./ expect(idOp, psi_t2)), label="Stochastic Schodenger Equation with record")
plot!(p, tlist, real.(expect(dagger(a) * a, psi_t)), label="Stochastic Schodenger Equation")

tend = 0.1
p = plot(tlist[tlist.<tend], real.(expect(idOp, psi_t2))[tlist.<tend], label="Stochastic Master Equation with Record", xlabel="Time", ylabel="normalization")
plot!(p, tlist[tlist.<tend], exp.(real.(logtrρ_t))[tlist.<tend], label="Stochastic Schodenger Equation")

p = plot(jump_t, jump_probs_master, label="Stochastic Master Equation with Record")
plot!(jump_t, jump_probs_sch, label="Stochastic Schodenger Equation")

glist = 2π * LinRange(35, 57, 30)
tlist = 0:0.0001:1
gacc = 2π * 45.0
start_t = 0.00
cumm_log_probs, max_likelihood_indices = master_param_est(glist=glist, tlist=tlist, gacc=gacc, ε=ε)
cumm_log_probs_sch, max_likelihood_indices_sch = hideo_param_est(glist=glist, tlist=tlist, gacc=gacc, ε=ε)


# 3D plot of the average probability of the different g's over time
surface(tlist[tlist.>=start_t], glist ./ (2π), exp.(cumm_log_probs[tlist.>=start_t, :]'), xlabel="Time (μs)", ylabel="g (MHz)", zlabel="Average Probability Density", label="Average Probability of g", colorbar=false, wireframe=true)


p = plot(tlist, map(i -> glist[i], max_likelihood_indices) ./ (2π), xlabel="Time (μs)", ylabel="g (MHz)", label="Master Eq Max Likelihood g")
plot!(p, tlist, map(i -> glist[i], max_likelihood_indices_sch) ./ (2π), xlabel="Time (μs)", ylabel="g (MHz)", label="Schodenger Eq Max Likelihood g")
title!(p, "Maximum Likelyhood Estimation Over a Single Trajectory", fontsize=16)
ylims!(p, glist[begin] / 2π, glist[end] / 2π)
hline!(p, [gacc / (2π)], linestyle=:dash, label="Ground Truth")

# same for stochastic Schodenger's equations
surface(tlist[tlist.>=start_t], glist ./ (2π), exp.(cumm_log_probs_sch[tlist.>=start_t, :]'), xlabel="Time (μs)", ylabel="g (MHz)", zlabel="Average Probability Density", label="Average Probability of g", colorbar=false, wireframe=true)
p = plot(tlist, map(i -> glist[i], max_likelihood_indices_sch) ./ (2π), xlabel="Time (μs)", ylabel="g (MHz)", label="Max Likelihood g")
ylims!(p, glist[begin] / 2π, glist[end] / 2π)
hline!(p, [gacc / (2π)], linestyle=:dash, label="Ground Truth")

p = plot(tlist, real.(expect(idOp, ρ_t)), label="Stochastic Master Equation with Record", xlabel="Time", ylabel="Photon Number")
plot!(p, tlist, real.(expect(idOp, psi_t2)), label="Stochastic Schodenger Equation with Record", xlabel="Time", ylabel="Photon Number")
# plot(jump_t, jump_probs, label="jump probabilites")

plot(tlist, map(x -> real.(norm(x)), ρ_t))

plot(norm.((ρ_t - dagger.(ρ_t)) / 2.0))