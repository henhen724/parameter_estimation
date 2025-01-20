using Base: AbstractArray, getindex, setindex!, size, map, similar, zero
using LinearAlgebra

# Define the Master Equation State
struct MasterEquationState{T<:Complex} <: AbstractArray{T,2}
    rho::Matrix{T}         # Matrix variable
    logtrrho::T            # A number
end


Base.:+(x::MasterEquationState, y::MasterEquationState) = MasterEquationState(x.rho + y.rho, x.logtrrho + y.logtrrho)
Base.:-(x::MasterEquationState, y::MasterEquationState) = MasterEquationState(x.rho - y.rho, x.logtrrho - y.logtrrho)
Base.:*(x::Number, y::MasterEquationState) = MasterEquationState(x * y.rho, x * y.logtrrho)
Base.:/(x::MasterEquationState, y::Number) = MasterEquationState(x.rho / y, x.logtrrho / y)

# Define the size of the MasterEquationState as the size of rho
function Base.size(s::MasterEquationState)
    return size(s.rho)
end

# Define how to access elements of the matrix (rho)
function Base.getindex(s::MasterEquationState, i::Integer, j::Integer)
    return s.rho[i, j]
end

# Define how to set elements of the matrix (rho)
function Base.setindex!(s::MasterEquationState, val, i::Integer, j::Integer)
    s.rho[i, j] = val
    return
end

# Implement length for the number of elements in rho, 
# though in the case of 2D, this may just be the number of 
# elements in the first dimension.
function Base.length(s::MasterEquationState)
    return length(s.rho)
end

# Add a method to retrieve logtrrho if needed
function getlogtrrho(s::MasterEquationState)
    return s.logtrrho
end

# Implement map functionality to apply a function to the rho matrix
function Base.map(f, s::MasterEquationState)
    new_rho = map(f, s.rho)
    return MasterEquationState(new_rho, s.logtrrho)
end

function Base.similar(s::MasterEquationState)
    return MasterEquationState(similar(s.rho), zero(s.logtrrho))
end

function Base.similar(s::MasterEquationState, ::Type{T}) where {T}
    return MasterEquationState(similar(s.rho, T), zero(T))
end

function Base.zero(s::MasterEquationState)
    return MasterEquationState(zero(s.rho), zero(s.logtrrho))
end

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