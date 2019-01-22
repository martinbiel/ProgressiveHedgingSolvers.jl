@with_kw mutable struct AsynchronousProgressiveHedgingData{T <: Real}
    Q::T = 1e10
    δ::T = 1.0
    δ₁::T = 1.0
    δ₂::T = 1.0
    timestamp::Int = 1
    iterations::Int = 0
end

@with_kw mutable struct AsynchronousProgressiveHedgingParameters{T <: Real}
    κ::T = 0.6
    r::T = 1.0
    τ::T = 1e-6
    log::Bool = true
end

"""
    AsynchronousProgressiveHedging

Functor object for the progressive-hedging algorithm. Create by supplying `:ph` to the `ProgressiveHedgingSolver` factory function and then pass to a `StochasticPrograms.jl` model, assuming there are available worker cores.

...
# Algorithm parameters
- `r::Real = 1.0`: Penalty parameter
- `τ::Real = 1e-6`: Relative tolerance for convergence checks.
- `log::Bool = true`: Specifices if progressive-hedging procedure should be logged on standard output or not.
...
"""
struct AsynchronousProgressiveHedging{T <: Real, A <: AbstractVector, SP <: StochasticProgram, S <: LQSolver} <: AbstractProgressiveHedgingSolver{T,A,S}
    stochasticprogram::SP
    solverdata::AsynchronousProgressiveHedgingData{T}

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
    parameters::AsynchronousProgressiveHedgingParameters{T}
    progress::ProgressThresh{T}

    @implement_trait AsynchronousProgressiveHedging Fixed
    @implement_trait AsynchronousProgressiveHedging Asynchronous

    function (::Type{AsynchronousProgressiveHedging})(stochasticprogram::StochasticProgram, x₀::AbstractVector, subsolver::QPSolver; kw...)
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
                           AsynchronousProgressiveHedgingData{T}(),
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
                           AsynchronousProgressiveHedgingParameters{T}(;kw...),
                           ProgressThresh(1.0, "Asynchronous Progressive Hedging "))
        # Initialize solver
        init!(ph, subsolver)
        return ph
    end
end
AsynchronousProgressiveHedging(stochasticprogram::StochasticProgram, subsolver::QPSolver; kw...) = AsynchronousProgressiveHedging(stochasticprogram, rand(decision_length(stochasticprogram)), subsolver; kw...)

function (ph::AsynchronousProgressiveHedging)()
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
