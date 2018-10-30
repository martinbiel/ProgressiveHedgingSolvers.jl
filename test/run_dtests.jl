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
using Gurobi

τ = 1e-5
reference_solver = GurobiSolver(OutputFlag=0)
dphsolvers = [(ProgressiveHedgingSolver(:dph,GurobiSolver(OutputFlag=0),log=false),"Progressive Hedging")]
phsolvers = [(ProgressiveHedgingSolver(:ph,GurobiSolver(OutputFlag=0),log=false),"Progressive Hedging")]

problems = Vector{Tuple{<:StochasticProgram,String}}()
@info "Loading test problems..."
@info "Loading simple..."
include("simple.jl")
@info "Loading farmer..."
include("farmer.jl")
@info "Loading infeasible..."
include("infeasible.jl")
@info "Loading integer..."
include("integer.jl")
@info "Test problems loaded. Starting test sequence."

@testset "Distributed solver" begin
    @testset "Distributed $phname Solver with Distributed Data: $name" for (phsolver,phname) in dphsolvers, (sp,name) in problems
        optimize!(sp, solver=reference_solver)
        x̄ = optimal_decision(sp)
        Q̄ = optimal_value(sp)
        optimize!(sp, solver=phsolver)
        @test abs(optimal_value(sp) - Q̄) <= τ*(1e-10+abs(Q̄))
        @test norm(optimal_decision(sp) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
    end
    @testset "Distributed $phname Solver: $name" for (phsolver,phname) in dphsolvers, (sp,name) in problems
        sp_nondist = copy(sp, procs=[1])
        optimize!(sp_nondist, solver=reference_solver)
        x̄ = optimal_decision(sp_nondist)
        Q̄ = optimal_value(sp_nondist)
        optimize!(sp_nondist, solver=phsolver)
        @test abs(optimal_value(sp_nondist) - Q̄) <= τ*(1e-10+abs(Q̄))
        @test norm(optimal_decision(sp) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
    end
    @testset "$phname Solver with Distributed Data: $name" for (phsolver,phname) in phsolvers, (sp,name) in problems
        optimize!(sp, solver=reference_solver)
        x̄ = optimal_decision(sp)
        Q̄ = optimal_value(sp)
        with_logger(NullLogger()) do
            optimize!(sp,solver=phsolver)
        end
        @test abs(optimal_value(sp) - Q̄) <= τ*(1e-10+abs(Q̄))
        @test norm(optimal_decision(sp) - x̄)/(1e-10+norm(x̄)) <= sqrt(τ)
    end
end
