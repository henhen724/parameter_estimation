"""
    mb(op, bases, idx)

Constructs a many-body operator by tensoring the given operator `op` with identity operators.

# Arguments
- `op`: The operator to be placed at position `idx`.
- `bases`: A list of bases for each Hilbert space.
- `idx`: The index at which to place the operator `op`.

# Returns
- `mbop`: The resulting many-body operator.

# Example
"""
function mb(op, bases, idx)

    numHilberts = size(bases, 1)

    if idx == 1
        mbop = op
    else
        mbop = identityoperator(bases[1])
    end

    for i = 2:numHilberts

        if i == idx
            mbop = tensor(mbop, op)
        else
            mbop = tensor(mbop, identityoperator(bases[i]))
        end

    end

    return mbop
end


"""
    integrate_schrodinger_with_record(H_nh, c_ops, tlist, ψ0, jump_t, jump_index)

Integrates the Schrödinger equation with non-Hermitian Hamiltonian `H_nh` and collapse operators `c_ops` over the time list `tlist`, starting from initial state `ψ0`. Records the jump probabilities at specified jump times `jump_t` and corresponding jump indices `jump_index`.

# Arguments
- `H_nh::AbstractMatrix`: Non-Hermitian Hamiltonian matrix.
- `c_ops::Vector{AbstractMatrix}`: Vector of collapse operators.
- `tlist::Vector{Float64}`: List of time points for integration.
- `ψ0::AbstractVector`: Initial state vector.
- `jump_t::Vector{Float64}`: List of times at which jumps occur.
- `jump_index::Vector{Int}`: Indices of collapse operators corresponding to each jump time.

# Returns
- `sol::ODESolution`: Solution object containing the time evolution of the state vector.
- `jump_probs::Vector{Float64}`: Vector of jump probabilities at the specified jump times.

# Example
"""
function integrate_schrodinger_with_record(H_nh, c_ops, tlist, ψ0, jump_t, jump_index)
    function schrodinger_rhs!(du, u, p, t)
        mul!(du, H_nh.data, u, -im, false)
    end
    jump_probs = zeros(length(jump_t))
    tmp_x = similar(ψ0.data)
    function dojump!(integrator)
        x = integrator.u
        t = integrator.t
        time_index = findmin(j_t -> abs(j_t - t), jump_t)[2]
        mul!(tmp_x, c_ops[jump_index[time_index]].data, x, 1.0, 0.0)
        tmp_norm = norm(tmp_x)
        if tmp_norm > eps(eltype(tmp_norm))
            integrator.u = tmp_x ./ tmp_norm
            jump_probs[time_index] = tmp_norm^2
        else
            jump_probs[time_index] = 0.0
        end
    end
    u0 = vec(ψ0.data)

    jump_cb = PresetTimeCallback(jump_t, dojump!, save_positions=(false, false))
    prob = ODEProblem(schrodinger_rhs!, u0, (tlist[1], tlist[end]), saveat=tlist)
    sol = solve(prob, Tsit5(); reltol=1e-5, abstol=1e-5, saveat=tlist, save_everystep=false, callback=jump_cb, tstops=jump_t)
    return sol, jump_probs
end