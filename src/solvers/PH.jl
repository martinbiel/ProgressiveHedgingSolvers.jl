@with_kw mutable struct ProgressiveHedgingData{T <: Real}
    Q::T = 1e10
    δ::T = 1.0
    δ₁::T = 1.0
    δ₂::T = 1.0
    iterations::Int = 0
end

@with_kw mutable struct ProgressiveHedgingParameters{T <: Real}
    r::T = 1.0
    τ::T = 1e-6
    log::Bool = true
end

"""
    ProgressiveHedging

Functor object for the progressive-hedging algorithm. Create by supplying `:ph` to the `ProgressiveHedgingSolver` factory function and then pass to a `StochasticPrograms.jl` model.

...
# Algorithm parameters
- `r::Real = 1.0`: Penalty parameter
- `τ::Real = 1e-6`: Relative tolerance for convergence checks.
- `log::Bool = true`: Specifices if progressive-hedging procedure should be logged on standard output or not.
...
"""
struct ProgressiveHedging{T <: Real, A <: AbstractVector, SP <: StochasticProgram, S <: LQSolver} <: AbstractProgressiveHedgingSolver{T,A,S}
    stochasticprogram::SP
    solverdata::ProgressiveHedgingData{T}

    # Estimate
    c::A
    ξ::A
    Q_history::A
    dual_gaps::A

    # Subproblems
    nscenarios::Int
    subproblems::Vector{SubProblem{T,A,S}}

    # Params
    parameters::ProgressiveHedgingParameters{T}
    progress::ProgressThresh{T}

    @implement_trait ProgressiveHedging Fixed

    function (::Type{ProgressiveHedging})(stochasticprogram::StochasticProgram, x₀::AbstractVector, subsolver::MPB.AbstractMathProgSolver; kw...)
        if nworkers() > 1
            @warn "There are worker processes, consider using distributed version of algorithm"
        end
        first_stage = StochasticPrograms.get_stage_one(stochasticprogram)
        length(x₀) != first_stage.numCols && error("Incorrect length of starting guess, has ", length(x₀), " should be ", first_stage.numCols)

        T = promote_type(eltype(x₀), Float32)
        c_ = convert(AbstractVector{T}, JuMP.prepAffObjective(first_stage))
        c_ *= first_stage.objSense == :Min ? 1 : -1
        x₀_ = convert(AbstractVector{T}, copy(x₀))
        A = typeof(x₀_)
        SP = typeof(stochasticprogram)
        S = LQSolver{typeof(MPB.LinearQuadraticModel(subsolver)), typeof(subsolver)}
        n = StochasticPrograms.nscenarios(stochasticprogram)

        ph = new{T,A,SP,S}(stochasticprogram,
                           ProgressiveHedgingData{T}(),
                           c_,
                           x₀_,
                           A(),
                           A(),
                           n,
                           Vector{SubProblem{T,A,S}}(),
                           ProgressiveHedgingParameters{T}(;kw...),
                           ProgressThresh(1.0, "Progressive Hedging"))
        # Initialize solver
        init!(ph, subsolver)
        return ph
    end
end
ProgressiveHedging(stochasticprogram::StochasticProgram, subsolver::MPB.AbstractMathProgSolver; kw...) = ProgressiveHedging(stochasticprogram, rand(decision_length(stochasticprogram)), subsolver; kw...)

function (ph::ProgressiveHedging)()
    # Reset timer
    ph.progress.tfirst = ph.progress.tlast = time()
    # Start procedure
    while true
        status = iterate!(ph)
        if status != :Valid
            return status
        end
    end
end
