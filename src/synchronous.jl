@implement_traitfn function resolve_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A}, Synchronous) where {T <: Real, A <: AbstractVector}
    partial_objectives = Vector{T}(undef, nworkers())
    @sync begin
        for (i,w) in enumerate(workers())
            @async partial_objectives[i] = remotecall_fetch(resolve_subproblems!, w, ph.subworkers[w-1], ph.ξ, penalty(ph))
        end
    end
    return sum(partial_objectives)
end

@implement_traitfn function update_iterate!(ph::AbstractProgressiveHedgingSolver{T,A}, Synchronous) where {T <: Real, A <: AbstractVector}
    partial_primals = Vector{A}(undef, nworkers())
    @sync begin
        for (i,w) in enumerate(workers())
            @async partial_primals[i] = remotecall_fetch(collect_primals, w, ph.subworkers[w-1], length(ph.ξ))
        end
    end
    ξ_prev = copy(ph.ξ)
    ph.ξ[:] = sum(partial_primals)
    # Update δ₁
    ph.solverdata.δ₁ = norm(ph.ξ-ξ_prev, 2)^2
end

@implement_traitfn function update_subproblems!(ph::AbstractProgressiveHedgingSolver, Synchronous)
    # Update dual prices
    @sync begin
        for w in workers()
            @async remotecall_fetch((sw,ξ,r)->begin
                subproblems = fetch(sw)
                if length(subproblems) > 0
                    update_subproblems!(subproblems, ξ, r)
                end
                end,
                w,
                ph.subworkers[w-1],
                ph.ξ,
                penalty(ph))
        end
    end
    return nothing
end

function resolve_subproblems!(subworker::SubWorker{T,A,S}, ξ::AbstractVector, r::AbstractFloat) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    Qs = A(undef, length(subproblems))
    for (i,subproblem) ∈ enumerate(subproblems)
        reformulate_subproblem!(subproblem, ξ, r)
        Qs[i] = subproblem()
    end
    return sum(Qs)
end

function collect_primals(subworker::SubWorker{T,A,S}, n::Integer) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    if length(subproblems) > 0
        return sum([subproblem.π*subproblem.x for subproblem in subproblems])
    else
        return zeros(T,n)
    end
end
