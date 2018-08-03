__precompile__()
module ProgressiveHedgingSolvers

using TraitDispatch
using Parameters
using JuMP
using StochasticPrograms
using MathProgBase
using RecipesBase
using ProgressMeter

import Base: show, put!, wait, isready, take!, fetch
import StochasticPrograms: StructuredModel, optimsolver, optimize_structured!, fill_solution!
importall MathProgBase.SolverInterface

export
    ProgressiveHedgingSolver,
    Crash,
    StructuredModel,
    optimsolver,
    optimize_structured!,
    fill_solution!,
    get_decision,
    get_objective_value

# Include files
include("LQSolver.jl")
include("subproblem.jl")
include("AbstractProgressiveHedging.jl")
include("parallel.jl")
include("solvers/ProgressiveHedging.jl")
#include("solvers/DProgressiveHedging.jl")
include("crash.jl")
include("spinterface.jl")

end # module
