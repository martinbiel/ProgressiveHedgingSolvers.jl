__precompile__()
module ProgressiveHedgingSolvers

# Standard library
using LinearAlgebra
using SparseArrays
using Distributed
using Printf

# External libraries
using TraitDispatch
using Parameters
using JuMP
using StochasticPrograms
using StochasticPrograms: _WS
using MathProgBase
using RecipesBase
using ProgressMeter

import Base: show, put!, wait, isready, take!, fetch
import StochasticPrograms: StructuredModel, internal_solver, optimize_structured!, fill_solution!, solverstr

const MPB = MathProgBase

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
include("synchronous.jl")
include("asynchronous.jl")
include("penalty.jl")
include("solvers/PH.jl")
include("solvers/AdaptivePH.jl")
include("solvers/SynchronousPH.jl")
include("solvers/SynchronousAdaptivePH.jl")
include("solvers/AsynchronousPH.jl")
include("solvers/AsynchronousAdaptivePH.jl")
include("crash.jl")
include("spinterface.jl")

end # module
