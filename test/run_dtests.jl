using Test
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

problems = Vector{Tuple{JuMP.Model,String}}()
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
@testset "Distributed $phname Solver with Distributed Data: $name" for (phsolver,phname) in dphsolvers, (sp,name) in problems
    solve(sp,solver=reference_solver)
    x̄ = copy(sp.colVal)
    Q̄ = copy(sp.objVal)
    solve(sp,solver=phsolver)
    @test abs(optimal_value(sp) - Q̄) <= τ*(1e-10+abs(Q̄))
end

@testset "Distributed $phname Solver: $name" for (phsolver,phname) in dphsolvers, (sp,name) in problems
    sp_nondist = StochasticProgram(first_stage_data(sp),second_stage_data(sp),scenarios(sp),procs=[1])
    transfer_model!(stochastic(sp_nondist),stochastic(sp))
    generate!(sp_nondist)
    solve(sp_nondist,solver=reference_solver)
    x̄ = copy(sp_nondist.colVal)
    Q̄ = copy(sp_nondist.objVal)
    solve(sp_nondist,solver=phsolver)
    @test abs(optimal_value(sp_nondist) - Q̄) <= τ*(1e-10+abs(Q̄))
end

@testset "$phname Solver with Distributed Data: $name" for (phsolver,phname) in phsolvers, (sp,name) in problems
    solve(sp,solver=reference_solver)
    x̄ = copy(sp.colVal)
    Q̄ = copy(sp.objVal)
    with_logger(NullLogger()) do
        solve(sp,solver=phsolver)
    end
    @test abs(optimal_value(sp) - Q̄) <= τ*(1e-10+abs(Q̄))
end
