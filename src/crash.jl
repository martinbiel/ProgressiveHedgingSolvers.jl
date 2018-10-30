"""
    Crash

Collection of crash methods used to generate initial decisions in L-shaped algorithms. Supply as functor object to the `crash` keyword in calls to `LShapedSolver`

...
# Crash methods
- `Crash.None()`: Randomize the initial decision (default).
- `Crash.EVP()`: Solve the expected value problem corresponding to the stochastic program and use the expected value solution as initial decision.
- `Crash.Scenario(scenario::AbstractScenario)`: Solve the wait-and-see problem corresponding a supplied scenario and use the optimal solution as initial decision.
- `Crash.Custom(x₀)`: Use the user-supplied `x₀` as initial decision.
...
"""
module Crash

using StochasticPrograms
using MathProgBase

abstract type CrashMethod end

struct None <: CrashMethod end

function (crash::None)(sp::StochasticProgram, solver::MathProgBase.AbstractMathProgSolver)
    return rand(StochasticPrograms.decision_length(sp))
end

struct EVP <: CrashMethod end

function (crash::EVP)(sp::StochasticProgram, solver::MathProgBase.AbstractMathProgSolver)
    evp = StochasticPrograms.EVP(sp; solver = solver)
    status = solve(evp)
    status != :Optimal && error("Could not solve EVP model during crash procedure. Aborting.")
    return evp.colVal[1:decision_length(sp)]
end

struct Scenario{S <: AbstractScenario} <: CrashMethod
    scenario::S

    function (::Type{Scenario})(scenario::S) where S <: AbstractScenario
        return new{S}(scenario)
    end
end

function (crash::Scenario)(so::StochasticProgram, solver::MathProgBase.AbstractMathProgSolver)
    ws = WS(sp, crash.scenario; solver = solver)
    status = solve(ws)
    status != :Optimal && error("Could not solve wait-and-see model during crash procedure. Aborting.")
    return ws.colVal[1:decision_length(sp)]
end

struct Custom{T <: Real} <: CrashMethod
    x₀::Vector{T}

    function (::Type{Custom})(x₀::Vector{T}) where T <: Real
        return new{T}(x₀)
    end
end

function (crash::Custom)(sp::StochasticProgram, solver::MathProgBase.AbstractMathProgSolver)
    return crash.x₀[1:decision_length(sp)]
end

end
