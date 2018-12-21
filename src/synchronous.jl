@implement_traitfn function resolve_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A}, Synchronous) where {T <: Real, A <: AbstractVector}
    active_workers = Vector{Future}(undef, nworkers())
    for w in workers()
        active_workers[w-1] = remotecall(resolve_subproblems!, w, ph.subworkers[w-1], ph.ξ, penalty(ph))
    end
    map(wait, active_workers)
    return sum(map(fetch, active_workers))
end

@implement_traitfn function update_iterate!(ph::AbstractProgressiveHedgingSolver, Synchronous)
    active_workers = Vector{Future}(undef, nworkers())
    for w in workers()
        active_workers[w-1] = remotecall(collect_primals, w, ph.subworkers[w-1], length(ph.ξ))
    end
    map(wait, active_workers)
    ξ_prev = copy(ph.ξ)
    ph.ξ[:] = sum(fetch.(active_workers))
    # Update δ₁
    ph.solverdata.δ₁ = norm(ph.ξ-ξ_prev, 2)^2
end

@implement_traitfn function update_subproblems!(ph::AbstractProgressiveHedgingSolver, Synchronous)
    active_workers = Vector{Future}(undef, nworkers())
    # Update dual prices
    for w in workers()
        active_workers[w-1] = remotecall((sw,ξ,r)->begin
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
    map(wait, active_workers)
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
