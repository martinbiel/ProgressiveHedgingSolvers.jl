@with_kw mutable struct DAdaptiveProgressiveHedgingData{T <: Real}
    Q::T = 1e10
    r::T = 1.0
    δ::T = 1.0
    δ₁::T = 1.0
    δ₂::T = 1.0
    iterations::Int = 0
end

@with_kw mutable struct DAdaptiveProgressiveHedgingParameters{T <: Real}
    ζ::T = 0.1
    γ₁::T = 1e-5
    γ₂::T = 0.01
    γ₃::T = 0.25
    σ::T = 1e-5
    α::T = 0.95
    θ::T = 1.1
    ν::T = 0.1
    β::T = 1.1
    η::T = 1.25
    τ::T = 1e-6
    log::Bool = true
end

"""
    DAdaptiveProgressiveHedging

Functor object for the progressive-hedging algorithm. Create by supplying `:ph` to the `ProgressiveHedgingSolver` factory function and then pass to a `StochasticPrograms.jl` model.

...
# Algorithm parameters
- `r::Real = 1.0`: Penalty parameter
- `τ::Real = 1e-6`: Relative tolerance for convergence checks.
- `log::Bool = true`: Specifices if progressive-hedging procedure should be logged on standard output or not.
...
"""
struct DAdaptiveProgressiveHedging{T <: Real, A <: AbstractVector, SP <: StochasticProgram, S <: LQSolver} <: AbstractProgressiveHedgingSolver{T,A,S}
    stochasticprogram::SP
    solverdata::DAdaptiveProgressiveHedgingData{T}

    # Estimate
    c::A
    ξ::A
    Q_history::A
    dual_gaps::A

    # Subproblems
    nscenarios::Int
    subworkers::Vector{SubWorker{T,A,S}}

    # Params
    parameters::DAdaptiveProgressiveHedgingParameters{T}
    progress::ProgressThresh{T}

    @implement_trait DAdaptiveProgressiveHedging Adaptive
    @implement_trait DAdaptiveProgressiveHedging Parallel

    function (::Type{DAdaptiveProgressiveHedging})(stochasticprogram::StochasticProgram, x₀::AbstractVector, subsolver::MPB.AbstractMathProgSolver; kw...)
        if nworkers() == 1
            @warn "There are no worker processes, defaulting to serial version of algorithm"
            return AdaptiveProgressiveHedging(stochasticprogram, x₀, subsolver; kw...)
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
                           DAdaptiveProgressiveHedgingData{T}(),
                           c_,
                           x₀_,
                           A(),
                           A(),
                           n,
                           Vector{SubWorker{T,A,S}}(undef, nworkers()),
                           DAdaptiveProgressiveHedgingParameters{T}(;kw...),
                           ProgressThresh(1.0, "Progressive Hedging"))
        # Initialize solver
        init!(ph, subsolver)
        return ph
    end
end
DAdaptiveProgressiveHedging(stochasticprogram::StochasticProgram, subsolver::MPB.AbstractMathProgSolver; kw...) = DAdaptiveProgressiveHedging(stochasticprogram, rand(decision_length(stochasticprogram)), subsolver; kw...)

function (ph::DAdaptiveProgressiveHedging)()
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
