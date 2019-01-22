"""
    ProgressiveHedgingSolver(qpsolver::AbstractMathProgSolver; <keyword arguments>)

Return a progressive-hedging algorithm object specified. Supply `qpsolver`, a MathProgBase solver capable of solving quadratic problems.

The following penalty parameter update procedures are available
- `:none`:  Fixed penalty (default) ?ProgressiveHedging for parameter descriptions.
- `:adaptive`: Adaptive penalty update ?AdaptiveProgressiveHedging for parameter descriptions.

The following execution policies are available
- `:sequential`:  Classical progressive-hedging (default) ?ProgressiveHedging for parameter descriptions.
- `:synchronous`: Classical progressive-hedging run in parallel ?SynchronousProgressiveHedging for parameter descriptions.
- `:asynchronous`: Asynchronous progressive-hedging ?AsynchronousPH for parameter descriptions.

...
# Arguments
- `qpsolver::AbstractMathProgSolver`: MathProgBase solver capable of solving quadratic programs.
- `penalty::Symbol = :none`: Specify penalty update procedure (:none, :adaptive)
- `execution::Symbol = :sequential`: Specify how algorithm should be executed (:sequential, :synchronous, :asynchronous). Distributed variants requires worker cores.
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
struct ProgressiveHedgingSolver{S <: QPSolver} <: AbstractStructuredSolver
    qpsolver::S
    penalty::Symbol
    execution::Symbol
    parameters

    function (::Type{ProgressiveHedgingSolver})(qpsolver::QPSolver; crash::Crash.CrashMethod = Crash.None(), penalty = :fixed, execution = :sequential, kwargs...)
        return new{typeof(qpsolver)}(qpsolver, penalty, execution, kwargs)
    end
end

function StructuredModel(stochasticprogram::StochasticProgram, solver::ProgressiveHedgingSolver)
    if solver.penalty == :fixed
        if solver.execution == :synchronous
            return SynchronousProgressiveHedging(stochasticprogram, solver.qpsolver; solver.parameters...)
        elseif solver.execution == :asynchronous
            return AsynchronousProgressiveHedging(stochasticprogram, solver.qpsolver; solver.parameters...)
        elseif solver.execution == :sequential
            return ProgressiveHedging(stochasticprogram, get_solver(solver.qpsolver); solver.parameters...)
        else
            error("Unknown execution: ", solver.execution)
        end
    elseif solver.penalty == :adaptive
        if solver.execution == :synchronous
            return SynchronousAdaptiveProgressiveHedging(stochasticprogram, solver.qpsolver; solver.parameters...)
        elseif solver.execution == :asynchronous
            return AsynchronousAdaptiveProgressiveHedging(stochasticprogram, solver.qpsolver; solver.parameters...)
        elseif solver.execution == :sequential
            return AdaptiveProgressiveHedging(stochasticprogram, get_solver(solver.qpsolver); solver.parameters...)
        else
            error("Unknown execution: ", solver.execution)
        end
    else
        error("Unknown penalty procedure: ", solver.penalty)
    end
end

function internal_solver(solver::ProgressiveHedgingSolver)
    return get_solver(solver.qpsolver)
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
