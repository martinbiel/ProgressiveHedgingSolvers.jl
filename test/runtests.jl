using ProgressiveHedgingSolvers
using Test
using Distributed
using JuMP
using StochasticPrograms
using Ipopt
using GLPKMathProgInterface

τ = 1e-5
reference_solver = GLPKSolverLP()
qp_solver = IpoptSolver(print_level=0)
phsolvers = [(ProgressiveHedgingSolver(:ph,qp_solver,log=false),"Progressive Hedging")]

problems = Vector{Tuple{JuMP.Model,String}}()
@info "Loading test problems..."
@info "Loading simple..."
include("simple.jl")
@info "Loading farmer..."
include("farmer.jl")
@info "Loading infeasible..."
include("infeasible.jl")
@info "Test problems loaded. Starting test sequence."

@testset "Sequential solver" begin
    @testset "$phname Solver: $name" for (phsolver,phname) in phsolvers, (sp,name) in problems
        solve(sp,solver=reference_solver)
        x̄ = optimal_decision(sp)
        Q̄ = optimal_value(sp)
        solve(sp,solver=phsolver)
        @test abs(optimal_value(sp) - Q̄) <= τ*(1e-10+abs(Q̄))
    end
end

@info "Starting distributed tests..."

include(joinpath(Sys.BINDIR, "..", "share", "julia", "test", "testenv.jl"))
disttestfile = joinpath(@__DIR__, "run_dtests.jl")
push!(test_exeflags.exec,"--color=yes")
cmd = `$test_exename $test_exeflags $disttestfile`

if !success(pipeline(cmd; stdout=stdout, stderr=stderr)) && ccall(:jl_running_on_valgrind,Cint,()) == 0
    error("Distributed test failed, cmd : $cmd")
end
