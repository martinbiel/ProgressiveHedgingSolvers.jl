using ProgressiveHedgingSolvers
using Test
using LinearAlgebra
using JuMP
using StochasticPrograms
using GLPKMathProgInterface
using OSQP

τ = 1e-2
reference_solver = GLPKSolverLP()
osqp = OSQP.OSQPMathProgBaseInterface.OSQPSolver(verbose=0)

penalties = [Fixed(),
             Adaptive(θ = 1.01)]

problems = Vector{Tuple{<:StochasticProgram,String}}()
@info "Loading test problems..."
@info "Loading simple..."
include("simple.jl")
@info "Loading farmer..."
include("farmer.jl")
@info "Loading infeasible..."
include("infeasible.jl")

@testset "Sequential solver" begin
    @testset "$(solverstr(ph)): $name" for ph in [ProgressiveHedgingSolver(osqp,
                                                                           penalty = penalty,
                                                                           τ = 1e-3,
                                                                           log = false)
                                                  for penalty in penalties], (sp,name) in problems
        optimize!(sp, solver=reference_solver)
        x̄ = optimal_decision(sp)
        Q̄ = optimal_value(sp)
        optimize!(sp, solver=ph)
        @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
        @test norm(optimal_decision(sp) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
    end
end

@info "Starting distributed tests..."

include("/usr/share/julia/test/testenv.jl")
push!(test_exeflags.exec,"--color=yes")
cmd = `$test_exename $test_exeflags run_dtests.jl`

if !success(pipeline(cmd; stdout=stdout, stderr=stderr)) && ccall(:jl_running_on_valgrind,Cint,()) == 0
    error("Distributed test failed, cmd : $cmd")
end
