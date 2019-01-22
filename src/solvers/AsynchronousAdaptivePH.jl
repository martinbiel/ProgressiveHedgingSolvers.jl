@with_kw mutable struct AsynchronousAdaptiveProgressiveHedgingData{T <: Real}
    Q::T = 1e10
    r::T = 1.0
    δ::T = 1.0
    δ₁::T = 1.0
    δ₂::T = 1.0
    timestamp::Int = 1
    iterations::Int = 0
end

@with_kw mutable struct AsynchronousAdaptiveProgressiveHedgingParameters{T <: Real}
    κ::T = 0.6
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
    AsynchronousAdaptiveProgressiveHedging

Functor object for the progressive-hedging algorithm. Create by supplying `:ph` to the `ProgressiveHedgingSolver` factory function and then pass to a `StochasticPrograms.jl` model, assuming there are available worker cores.

...
# Algorithm parameters
- `r::Real = 1.0`: Penalty parameter
- `τ::Real = 1e-6`: Relative tolerance for convergence checks.
- `log::Bool = true`: Specifices if progressive-hedging procedure should be logged on standard output or not.
...
"""
struct AsynchronousAdaptiveProgressiveHedging{T <: Real, A <: AbstractVector, SP <: StochasticProgram, S <: LQSolver} <: AbstractProgressiveHedgingSolver{T,A,S}
    stochasticprogram::SP
    solverdata::AsynchronousAdaptiveProgressiveHedgingData{T}

    # Estimate
    c::A
    ξ::A
    Q_history::A
    dual_gaps::A

    # Subproblems
    nscenarios::Int
    subobjectives::Vector{A}
    finished::Vector{Int}

    # Workers
    subworkers::Vector{SubWorker{T,A,S}}
    work::Vector{Work}
    progressqueue::ProgressQueue{T}
    x̄::Vector{RunningAverage{A}}
    δ::Vector{RunningAverage{T}}
    decisions::Decisions{A}
    r::IteratedValue{T}
    active_workers::Vector{Future}

    # Params
    parameters::AsynchronousAdaptiveProgressiveHedgingParameters{T}
    progress::ProgressThresh{T}

    @implement_trait AsynchronousAdaptiveProgressiveHedging Adaptive
    @implement_trait AsynchronousAdaptiveProgressiveHedging Asynchronous

    function (::Type{AsynchronousAdaptiveProgressiveHedging})(stochasticprogram::StochasticProgram, x₀::AbstractVector, subsolver::QPSolver; kw...)
        if nworkers() == 1
            @warn "There are no worker processes, defaulting to serial version of algorithm"
            return ProgressiveHedging(stochasticprogram, x₀, get_solver(subsolver); kw...)
        end
        first_stage = StochasticPrograms.get_stage_one(stochasticprogram)
        length(x₀) != first_stage.numCols && error("Incorrect length of starting guess, has ", length(x₀), " should be ", first_stage.numCols)

        T = promote_type(eltype(x₀), Float32)
        c_ = convert(AbstractVector{T}, JuMP.prepAffObjective(first_stage))
        c_ *= first_stage.objSense == :Min ? 1 : -1
        x₀_ = convert(AbstractVector{T}, copy(x₀))
        A = typeof(x₀_)
        SP = typeof(stochasticprogram)
        solver_instance = get_solver(subsolver)
        S = LQSolver{typeof(MPB.LinearQuadraticModel(solver_instance)),typeof(solver_instance)}
        n = StochasticPrograms.nscenarios(stochasticprogram)

        ph = new{T,A,SP,S}(stochasticprogram,
                           AsynchronousAdaptiveProgressiveHedgingData{T}(),
                           c_,
                           x₀_,
                           A(),
                           A(),
                           n,
                           Vector{A}(),
                           Vector{Int}(),
                           Vector{SubWorker{T,A,S}}(undef, nworkers()),
                           Vector{Work}(undef,nworkers()),
                           RemoteChannel(() -> Channel{Progress{T}}(4*nworkers()*n)),
                           Vector{RunningAverage{A}}(undef,nworkers()),
                           Vector{RunningAverage{T}}(undef,nworkers()),
                           RemoteChannel(() -> IterationChannel(Dict{Int,A}())),
                           RemoteChannel(() -> IterationChannel(Dict{Int,T}())),
                           Vector{Future}(undef,nworkers()),
                           AsynchronousAdaptiveProgressiveHedgingParameters{T}(;kw...),
                           ProgressThresh(1.0, "Asynchronous Progressive Hedging "))
        # Initialize solver
        init!(ph, subsolver)
        return ph
    end
end
AsynchronousAdaptiveProgressiveHedging(stochasticprogram::StochasticProgram, subsolver::QPSolver; kw...) = AsynchronousAdaptiveProgressiveHedging(stochasticprogram, rand(decision_length(stochasticprogram)), subsolver; kw...)

function (ph::AsynchronousAdaptiveProgressiveHedging)()
    # Reset timer
    ph.progress.tfirst = ph.progress.tlast = time()
    # Start workers
    init_workers!(ph)
    # Start procedure
    while true
        status = iterate!(ph)
        if status != :Valid
            close_workers!(ph)
            return status
        end
    end
end
