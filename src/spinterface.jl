"""
    ProgressiveHedgingSolver(variant::Symbol = :ph, qpsolver::AbstractMathProgSolver; <keyword arguments>)

Return the progressive-hedging algorithm object specified by the `variant` symbol. Supply `qpsolver`, a MathProgBase solver capable of solving quadratic problems.

The available algorithm variants are as follows
- `:ph`:  Progressive-hedging algorithm (default) ?ProgressiveHedging for parameter descriptions.
- `:dph`: Distributed progressive-hedging algorithm (requires worker cores) ?DProgressiveHedging for parameter descriptions.

...
# Arguments
- `variant::Symbol = :ph`: progressive-hedging algorithm variant.
- `qpsolver::AbstractMathProgSolver`: MathProgBase solver capable of solving quadratic programs.
- `crash::Crash.CrashMethod = Crash.None`: Crash method used to generate an initial decision. See ?Crash for alternatives.
- <keyword arguments>: Algorithm specific parameters, consult individual docstrings (see above list) for list of possible arguments and default values.
...

## Examples

The following solves a stochastic program `sp` created in `StochasticPrograms.jl` using the progressive-hedging algorithm with Ipopt as an `qpsolver`.

```jldoctest
julia> solve(sp,solver=ProgressiveHedgingSolver(:ph, IpoptSolver(print_level=0)))
Progressive Hedging Time: 0:00:06 (1315 iterations)
  Objective:  -855.8332803469432
  δ:          9.436947935542464e-7
:Optimal
```
"""
struct ProgressiveHedgingSolver <: AbstractStructuredSolver
    variant::Symbol
    qpsolver::MPB.AbstractMathProgSolver
    crash::Crash.CrashMethod
    parameters

    function (::Type{ProgressiveHedgingSolver})(variant::Symbol, qpsolver::MPB.AbstractMathProgSolver; crash::Crash.CrashMethod = Crash.None(), kwargs...)
        return new(variant, qpsolver, crash, kwargs)
    end
end
ProgressiveHedgingSolver(subsolver::MPB.AbstractMathProgSolver; kwargs...) = ProgressiveHedgingSolver(:ph, subsolver, kwargs...)

function StructuredModel(stochasticprogram::StochasticProgram, solver::ProgressiveHedgingSolver)
    x₀ = solver.crash(stochasticprogram,solver.qpsolver)
    if solver.variant == :ph
        return ProgressiveHedging(stochasticprogram, x₀, solver.qpsolver; solver.parameters...)
    elseif solver.variant == :dph
        return DProgressiveHedging(stochasticprogram, x₀, solver.qpsolver; solver.parameters...)
    else
        error("Unknown progressive hedging variant: ", solver.variant)
    end
end

function internal_solver(solver::ProgressiveHedgingSolver)
    return solver.qpsolver
end

function optimize_structured!(ph::AbstractProgressiveHedgingSolver)
    return ph()
end

function fill_solution!(stochasticprogram::StochasticProgram, ph::AbstractProgressiveHedgingSolver)
    # First stage
    first_stage = StochasticPrograms.get_stage_one(stochasticprogram)
    nrows, ncols = first_stage_dims(stochasticprogram)
    StochasticPrograms.set_decision!(stochasticprogram, ph.ξ)
    # stochasticprogram.redCosts = try
    #     getreducedcosts(ph.mastersolver.lqmodel)[1:ncols]
    # catch
    #     fill(NaN, ncols)
    # end
    # stochasticprogram.linconstrDuals = try
    #     getconstrduals(ph.mastersolver.lqmodel)[1:nrows]
    # catch
    #     fill(NaN, nrows)
    # end
    # Second stage
    fill_submodels!(ph, scenarioproblems(stochasticprogram))
end

function solverstr(solver::ProgressiveHedgingSolver)
    if solver.variant == :ph
        return "Progressive-hedging"
    elseif solver.variant == :dph
        return "Distributed progressive-hedging"
    else
        error("Unknown progressive-hedging variant: ", solver.variant)
    end
end
