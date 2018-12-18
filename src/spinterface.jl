"""
    ProgressiveHedgingSolver(qpsolver::AbstractMathProgSolver; <keyword arguments>)

Return a progressive-hedging algorithm object specified. Supply `qpsolver`, a MathProgBase solver capable of solving quadratic problems.

The following penalty parameter update procedures are available
- `:none`:  Fixed penalty (default) ?ProgressiveHedging for parameter descriptions.
- `:adaptive`: Adaptive penalty update ?AdaptiveProgressiveHedging for parameter descriptions.

...
# Arguments
- `qpsolver::AbstractMathProgSolver`: MathProgBase solver capable of solving quadratic programs.
- `crash::Crash.CrashMethod = Crash.None`: Crash method used to generate an initial decision. See ?Crash for alternatives.
- `penalty::Symbol = :none`: Specify penalty update procedure (:none, :adaptive)
- `distributed::Bool = false`: Specify if distributed variant of algorithm should be run (requires worker cores). See `?Alg` for parameter descriptions.
- <keyword arguments>: Algorithm specific parameters, consult individual docstrings (see above list) for list of possible arguments and default values.
...

## Examples

The following solves a stochastic program `sp` created in `StochasticPrograms.jl` using the progressive-hedging algorithm with Ipopt as an `qpsolver`.

```jldoctest
julia> solve(sp,solver=ProgressiveHedgingSolver(IpoptSolver(print_level=0)))
Progressive Hedging Time: 0:00:06 (1315 iterations)
  Objective:  -855.8332803469432
  δ:          9.436947935542464e-7
:Optimal
```
"""
struct ProgressiveHedgingSolver <: AbstractStructuredSolver
    qpsolver::MPB.AbstractMathProgSolver
    crash::Crash.CrashMethod
    penalty::Symbol
    distributed::Bool
    parameters

    function (::Type{ProgressiveHedgingSolver})(qpsolver::MPB.AbstractMathProgSolver; crash::Crash.CrashMethod = Crash.None(), penalty = :fixed, distributed = false, kwargs...)
        return new(qpsolver, crash, penalty, distributed, kwargs)
    end
end

function StructuredModel(stochasticprogram::StochasticProgram, solver::ProgressiveHedgingSolver)
    x₀ = solver.crash(stochasticprogram,solver.qpsolver)
    if solver.penalty == :fixed
        if solver.distributed
            return DProgressiveHedging(stochasticprogram, x₀, solver.qpsolver; solver.parameters...)
        else
            return ProgressiveHedging(stochasticprogram, x₀, solver.qpsolver; solver.parameters...)
        end
    elseif solver.penalty == :adaptive
        if solver.distributed
            return DAdaptiveProgressiveHedging(stochasticprogram, x₀, solver.qpsolver; solver.parameters...)
        else
            return AdaptiveProgressiveHedging(stochasticprogram, x₀, solver.qpsolver; solver.parameters...)
        end
    else
        error("Unknown progressive hedging penalty: ", solver.penalty)
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
