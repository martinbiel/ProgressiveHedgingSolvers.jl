struct ProgressiveHedgingSolver <: AbstractStructuredSolver
    variant::Symbol
    subsolver::MPB.AbstractMathProgSolver
    crash::Crash.CrashMethod
    parameters

    function (::Type{ProgressiveHedgingSolver})(variant::Symbol, subsolver::MPB.AbstractMathProgSolver; crash::Crash.CrashMethod = Crash.None(), kwargs...)
        return new(variant,subsolver,crash,kwargs)
    end
end
ProgressiveHedgingSolver(subsolver::MPB.AbstractMathProgSolver; kwargs...) = ProgressiveHedgingSolver(:ph, subsolver, kwargs...)

function StructuredModel(solver::ProgressiveHedgingSolver,stochasticprogram::JuMP.Model)
    x₀ = solver.crash(stochasticprogram,solver.subsolver)
    if solver.variant == :ph
        return ProgressiveHedging(stochasticprogram,x₀,solver.subsolver; solver.parameters...)
    elseif solver.variant == :dph
        return DProgressiveHedging(stochasticprogram,x₀,solver.subsolver; solver.parameters...)
    else
        error("Unknown progressive hedging variant: ", solver.variant)
    end
end

function optimsolver(solver::ProgressiveHedgingSolver)
    return solver.subsolver
end

function optimize_structured!(ph::AbstractProgressiveHedgingSolver)
    return ph()
end

function fill_solution!(ph::AbstractProgressiveHedgingSolver,stochasticprogram::JuMP.Model)
    # First stage
    nrows, ncols = length(stochasticprogram.linconstr), stochasticprogram.numCols
    stochasticprogram.colVal = copy(ph.ξ)
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
    fill_submodels!(ph,scenarioproblems(stochasticprogram))
    # Now safe to generate the objective value of the stochastic program
    stochasticprogram.objVal = calculate_objective_value(stochasticprogram)
end
