struct FIMasterState{HS_dim,param_num}
    state::Vector{ComplexF64}

    function FIMasterState(state::Vector{ComplexF64})
        @assert length(state) == 1 + (1 + param_num) * HS_dim^2
        new{HS_dim,param_num}(state)
    end
end

function to_state!(state::FIMasterState, trρ, ρ_vec)
    state.state[1] = trρ
    state.state[2:end] .= ρ_vec
end

function to_state_alloc!(trρ, ρ_vec)
    state = FIMasterState(vcat([trρ], ρ_vec))
    return state
end

function logtrρ_from_state!(state::FIMasterState)
    return state.state[1]
end

function ρ_from_state!(state::FIMasterState, HS_dim)
    return reshape(state.state[2:end], (HS_dim, HS_dim))
end

function integrate_FI_master_equation(Hs, params, c_ops, ρ0, tlist, jump_t, jump_index)
    num_params = length(Hs)
    Hs_data = []
    for H in Hs
        push!(Hs_data, H.data)
    end
    @assert num_params == length(params)
    HS_dim = size(ρ0, 1)
    function to_state!(state, trρ, ρ_vec, par_ρ_list)
        state[1] = trρ
        state[2:HS_dim^2+1] .= ρ_vec
        for i in 1:num_params
            state[i*HS_dim^2+1:(i+1)*HS_dim^2+1] .= par_ρ_list[i]
        end
    end
    function to_state_alloc!(trρ, ρ_vec)
        state = zeros(ComplexF64, 1 + HS_dim^2)
        state[1] = trρ
        state[2:HS_dim^2+1] .= ρ_vec
        return state
    end
    function ρ_from_state!(state)
        return reshape(state[2:HS_dim^2+1], size(ρ0))
    end
    function logtrρ_from_state!(state)
        return state[1]
    end

    c_ops_data = map(x -> x.data, c_ops)
    function lindblad_rhs!(dstate, state, p, t)
        ρ = ρ_from_state!(state)
        dlogtrρ = 0.0im
        dρ = similar(ρ)
        for (idx, H_data) in enumerate(Hs_data)
            dρ .= -im * params[idx] * (H_data * ρ - ρ * H_data)
        end
        for c_op in c_ops_data
            dρ .+= tr(adjoint(c_op) * c_op * ρ) * ρ - 0.5 * (adjoint(c_op) * c_op * ρ + ρ * adjoint(c_op) * c_op)
            dlogtrρ = -tr(adjoint(c_op) * c_op * ρ) / tr(ρ)
        end
        to_state!(dstate, dlogtrρ, vec(dρ))
    end

    jump_probs = zeros(length(jump_t))
    tmp_x = similar(ρ0.data)
    function dojump!(integrator)
        x = integrator.u
        t = integrator.t
        time_index = findmin(j_t -> abs(j_t - t), jump_t)[2]
        ρ = ρ_from_state!(state)
        tmp_x = c_ops[jump_index[time_index]].data * ρ * dagger(c_ops[jump_index[time_index]]).data
        # print(tr(tmp_x))
        tmp_norm = real.(tr(tmp_x))
        if tmp_norm > eps(eltype(tmp_norm))
            to_state!(integrator.u, 0.0, vec(tmp_x ./ tmp_norm))
        else
            to_state!(integrator.u, 0.0, vec(tmp_x))
            jump_probs[time_index] = 0.0
        end
    end

    jump_cb = PresetTimeCallback(jump_t, dojump!, save_positions=(false, false))

    # Convert initial density matrix to vector form
    ρ0_vec = vec(ρ0.data)

    # Define the ODE problem
    prob = ODEProblem(lindblad_rhs!, to_state_alloc!(0.0, ρ0_vec), (tlist[1], tlist[end]), saveat=tlist)

    # Solve the ODE problem
    sol = solve(prob, Tsit5(); reltol=1e-3, abstol=1e-3, saveat=tlist, save_everystep=false, callback=jump_cb, tstops=jump_t)

    println("Solution length: ", length(sol))

    # Convert solution back to density matrix form
    ρ_t = [Operator(basis(ρ0), ρ_from_state!(sol.u[i])) for i in 1:length(sol.u)]
    logtrρ_t = [sol.u[i][1] for i in 1:length(sol.u)]

    return tlist, ρ_t, logtrρ_t
end