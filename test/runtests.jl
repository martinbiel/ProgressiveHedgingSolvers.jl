using ProgressiveHedgingSolvers
using Test
using LinearAlgebra
using JuMP
using StochasticPrograms
using Gurobi

τ = 1e-5
reference_solver = GurobiSolver(OutputFlag=0)
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

@testset "Sequential solver" begin
    @testset "$phname Solver: $name" for (phsolver,phname) in phsolvers, (sp,name) in problems
        optimize!(sp,solver=reference_solver)
        x̄ = optimal_decision(sp)
        Q̄ = optimal_value(sp)
        optimize!(sp,solver=phsolver)
        @test abs(optimal_value(sp) - Q̄) <= τ*(1e-10+abs(Q̄))
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
