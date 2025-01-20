using TestSetExtensions, LinearAlgebra
include("../states/index.jl")

@testset "States Tests" begin
    # Define a matrix rho and a value for logtrrho
    rho = Matrix{ComplexF32}([0.5 0.5; 0.5 0.5])
    logtrrho = log(tr(rho))  # Just an example of how to set logtrrho

    # Instantiate the custom state
    state = MasterEquationState(rho, logtrrho)

    @test rho == state.rho
    @test state[1, 1] == 0.5

    # Modifying an element
    state[1, 1] = 0.8
    @test state.rho[1, 1] == ComplexF32(0.8)

    # Using map (if implemented)
    new_state = map(x -> x + 1, state)
    @test new_state[1, 2] == 1.5
end

@testset "Conversion Tests" begin
    using QuantumOptics

    # Define a QuantumOptics state
    basis = FockBasis(2)
    qo_state = fockstate(basis, 1)
    rho_qo = dm(qo_state)

    # Convert to MasterEquationState
    logtrrho_qo = log(tr(rho_qo))
    me_state = MasterEquationState(rho_qo.data, logtrrho_qo)

    @test me_state.rho == rho_qo.data
    fk0 = fockstate(basis, 0)
    @test me_state[1, 1] == dagger(fk0) * rho_qo * fk0

    # Convert back to QuantumOptics state
    rho_converted = DenseOperator(basis, me_state.rho)
    @test rho_converted == rho_qo
end
