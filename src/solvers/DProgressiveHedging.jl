@with_kw mutable struct DProgressiveHedgingData{T <: Real}
    Q::T = 1e10
    δ::T = 1.0
    iterations::Int = 0
end

@with_kw mutable struct DProgressiveHedgingParameters{T <: Real}
    r::T = 1.0
    τ::T = 1e-6
    log::Bool = true
end

struct DProgressiveHedging{T <: Real, A <: AbstractVector, S <: LQSolver} <: AbstractProgressiveHedgingSolver{T,A,S}
    structuredmodel::JuMP.Model
    solverdata::DProgressiveHedgingData{T}

    # Estimate
    c::A
    ξ::A
    ρ::A
    Q_history::A

    # Workers
    nscenarios::Int
    subworkers::Vector{SubWorker{T,A,S}}

    # Params
    parameters::DProgressiveHedgingParameters{T}
    progress::ProgressThresh{T}

    @implement_trait DProgressiveHedging IsParallel

    function (::Type{DProgressiveHedging})(model::JuMP.Model,x₀::AbstractVector,subsolver::MPB.AbstractMathProgSolver; kw...)
        if nworkers() == 1
            @warn "There are no worker processes, defaulting to serial version of algorithm"
            return ProgressiveHedging(model,x₀,subsolver; kw...)
        end
        length(x₀) != model.numCols && error("Incorrect length of starting guess, has ",length(x₀)," should be ",model.numCols)
        !haskey(model.ext,:SP) && error("The provided model is not structured")

        T = promote_type(eltype(x₀),Float32)
        c_ = convert(AbstractVector{T},JuMP.prepAffObjective(model))
        c_ *= model.objSense == :Min ? 1 : -1
        mastervector = convert(AbstractVector{T},copy(x₀))
        x₀_ = convert(AbstractVector{T},copy(x₀))
        A = typeof(x₀_)

        S = LQSolver{typeof(MPB.LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(model)

        ph = new{T,A,S}(model,
                        DProgressiveHedgingData{T}(),
                        c_,
                        x₀_,
                        zero(x₀_),
                        A(),
                        n,
                        Vector{SubWorker{T,A,S}}(undef, nworkers()),
                        DProgressiveHedgingParameters{T}(;kw...),
                        ProgressThresh(1.0, "Distributed Progressive Hedging "))
        # Initialize solver
        init!(ph,subsolver)
        return ph
    end
end
DProgressiveHedging(model::JuMP.Model,subsolver::MPB.AbstractMathProgSolver; kw...) = DProgressiveHedging(model,rand(model.numCols),subsolver; kw...)

function (ph::DProgressiveHedging)()
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
