@with_kw mutable struct ProgressiveHedgingData{T <: Real}
    Q::T = 1e10
    δ::T = 1.0
    iterations::Int = 0
end

@with_kw mutable struct ProgressiveHedgingParameters{T <: Real}
    r::T = 1.0
    τ::T = 1e-6
    log::Bool = true
end

struct ProgressiveHedging{T <: Real, A <: AbstractVector, S <: LQSolver} <: AbstractProgressiveHedgingSolver{T,A,S}
    structuredmodel::JuMP.Model
    solverdata::ProgressiveHedgingData{T}

    # Estimate
    c::A
    ξ::A
    ρ::A
    Q_history::A

    # Subproblems
    nscenarios::Int
    subproblems::Vector{SubProblem{T,A,S}}
    subobjectives::A

    # Params
    parameters::ProgressiveHedgingParameters{T}
    progress::ProgressThresh{T}

    function (::Type{ProgressiveHedging})(model::JuMP.Model,x₀::AbstractVector,subsolver::AbstractMathProgSolver; kw...)
        if nworkers() > 1
            warn("There are worker processes, consider using distributed version of algorithm")
        end
        length(x₀) != model.numCols && error("Incorrect length of starting guess, has ",length(x₀)," should be ",model.numCols)
        !haskey(model.ext,:SP) && error("The provided model is not structured")

        T = promote_type(eltype(x₀),Float32)
        c_ = convert(AbstractVector{T},JuMP.prepAffObjective(model))
        c_ *= model.objSense == :Min ? 1 : -1
        x₀_ = convert(AbstractVector{T},copy(x₀))
        A = typeof(x₀_)

        S = LQSolver{typeof(LinearQuadraticModel(subsolver)),typeof(subsolver)}
        n = StochasticPrograms.nscenarios(model)

        ph = new{T,A,S}(model,
                        ProgressiveHedgingData{T}(),
                        c_,
                        x₀_,
                        zeros(x₀_),
                        A(),
                        n,
                        Vector{SubProblem{T,A,S}}(),
                        A(zeros(n)),
                        ProgressiveHedgingParameters{T}(;kw...),
                        ProgressThresh(1.0, "L-Shaped Gap "))
        # Initialize solver
        init!(ph,subsolver)
        return ph
    end
end
ProgressiveHedging(model::JuMP.Model,subsolver::AbstractMathProgSolver; kw...) = ProgressiveHedging(model,rand(model.numCols),subsolver; kw...)

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
