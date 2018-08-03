using ProgressiveHedgingSolvers
using Base.Test
using JuMP
using StochasticPrograms
using Gurobi

τ = 1e-5
reference_solver = GurobiSolver(OutputFlag=0)
phsolvers = [(ProgressiveHedgingSolver(:ph,GurobiSolver(OutputFlag=0),log=false),"Progressive Hedging"),
             (ProgressiveHedgingSolver(:ph,GurobiSolver(OutputFlag=0),log=false,linearize=true),"Linearized Progressive Hedging")]

problems = Vector{Tuple{JuMP.Model,String}}()
info("Loading test problems...")
info("Loading simple...")
include("simple.jl")
info("Loading farmer...")
include("farmer.jl")
info("Loading day-ahead problems...")
include("dayahead.jl")

info("Test problems loaded. Starting test sequence.")
@testset "$phname Solver: $name" for (phsolver,phname) in phsolvers, (sp,name) in problems
    solve(sp,solver=reference_solver)
    x̄ = optimal_decision(sp)
    Q̄ = optimal_value(sp)
    solve(sp,solver=phsolver)
    @test abs(optimal_value(sp) - Q̄) <= τ*(1e-10+abs(Q̄))
end

# info("Starting distributed tests...")

# include("/opt/julia-0.6/share/julia/test/testenv.jl")
# push!(test_exeflags.exec,"--color=yes")
# cmd = `$test_exename $test_exeflags run_dtests.jl`

# if !success(pipeline(cmd; stdout=STDOUT, stderr=STDERR)) && ccall(:jl_running_on_valgrind,Cint,()) == 0
#     error("Distributed test failed, cmd : $cmd")
# end
