using Base: AbstractArray, getindex, setindex!, size, map
using LinearAlgebra

# Define the Master Equation State
struct MasterEquationState{T<:Complex}
    rho::Matrix{T}         # Matrix variable
    logtrrho::T            # A number
end

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

# Example usage:
function main()
    # Define a matrix rho and a value for logtrrho
    rho = Matrix{ComplexF32}([0.5 0.5; 0.5 0.5])
    logtrrho = log(tr(rho))  # Just an example of how to set logtrrho

    # Instantiate the custom state
    state = MasterEquationState(rho, logtrrho)

    println("Initial State (rho):")
    println(state.rho)
    println("logtrrho: ", state.logtrrho)

    # Accessing elements
    println("Element (1,1): ", state[1, 1])

    # Modifying an element
    state[1, 1] = 0.8
    println("Modified State (rho):")
    println(state.rho)

    # Using map (if implemented)
    new_state = map(x -> x + 1, state)
    println("Mapped State (rho):")
    println(new_state.rho)
end

main()