using Test
using LinearAlgebra
using Distributed
using Logging
include("/usr/share/julia/test/testenv.jl")
addprocs_with_testenv(3)
@test nworkers() == 3

@everywhere using StochasticPrograms
using ProgressiveHedgingSolvers
using JuMP
using GLPKMathProgInterface
using OSQP

τ = 1e-2
reference_solver = GLPKSolverLP()
osqp = OSQP.OSQPMathProgBaseInterface.OSQPSolver(verbose=0)

executors = [Serial(),
             Synchronous()]

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
@info "Test problems loaded. Starting test sequence."

@testset "Distributed solver" begin
    @testset "$(solverstr(ph)): $name" for ph in [ProgressiveHedgingSolver(osqp,
                                                                           execution = executor,
                                                                           penalty = penalty,
                                                                           τ = 1e-3,
                                                                           log = false)
                                                  for executor in executors, penalty in penalties], (sp,name) in problems
        @testset "Distributed data" begin
            optimize!(sp, solver=reference_solver)
            x̄ = optimal_decision(sp)
            Q̄ = optimal_value(sp)
            with_logger(NullLogger()) do
                optimize!(sp, solver=ph)
            end
            @test abs(optimal_value(sp) - Q̄)/(1e-10+abs(Q̄)) <= τ
            @test norm(optimal_decision(sp) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
        end
        @testset "Data on single remote node" begin
            sp_onenode = copy(sp)
            add_scenarios!(sp_onenode, scenarios(sp), workers()[1])
            optimize!(sp_onenode, solver=reference_solver)
            x̄ = optimal_decision(sp_onenode)
            Q̄ = optimal_value(sp_onenode)
            with_logger(NullLogger()) do
                optimize!(sp, solver=ph)
            end
            @test abs(optimal_value(sp_onenode) - Q̄)/(1e-10+abs(Q̄)) <= τ
            @test norm(optimal_decision(sp_onenode) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
        end
        @testset "Local data" begin
            sp_nondist = copy(sp, procs = [1])
            add_scenarios!(sp_nondist, scenarios(sp))
            optimize!(sp_nondist, solver=reference_solver)
            x̄ = optimal_decision(sp_nondist)
            Q̄ = optimal_value(sp_nondist)
            with_logger(NullLogger()) do
                optimize!(sp, solver=ph)
            end
            @test abs(optimal_value(sp_nondist) - Q̄)/(1e-10+abs(Q̄)) <= τ
            @test norm(optimal_decision(sp_nondist) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
        end
    end
end
