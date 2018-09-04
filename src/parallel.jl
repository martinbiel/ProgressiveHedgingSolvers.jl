# ------------------------------------------------------------
# IsParallel -> Algorithm is run in parallel
# ------------------------------------------------------------
@define_trait IsParallel

@define_traitfn IsParallel init_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A,S},subsolver::AbstractMathProgSolver) where {T <: Real, A <: AbstractVector, S <: LQSolver} = begin
    function init_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A,S},subsolver::AbstractMathProgSolver,!IsParallel) where {T <: Real, A <: AbstractVector, S <: LQSolver}
        # Prepare the subproblems
        m = ph.structuredmodel
        load_subproblems!(ph,subsolver)
        ph.ξ[:] = sum([subproblem.π*subproblem.x for subproblem in ph.subproblems])
        return ph
    end

    function init_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A,S},subsolver::AbstractMathProgSolver,IsParallel) where {T <: Real, A <: AbstractVector, S <: LQSolver}
        # Partitioning
        (jobsize,extra) = divrem(ph.nscenarios,nworkers())
        # One extra to guarantee coverage
        if extra > 0
            jobsize += 1
        end
        # Create subproblems on worker processes
        m = ph.structuredmodel
        start = 1
        stop = jobsize
        active_workers = Vector{Future}(nworkers())
        for w in workers()
            ph.subworkers[w-1] = RemoteChannel(() -> Channel{Vector{SubProblem{T,A,S}}}(1), w)
            active_workers[w-1] = load_worker!(scenarioproblems(m),m,w,ph.subworkers[w-1],ph.ξ,start,stop,ph.parameters.r,subsolver)
            if start > ph.nscenarios
                continue
            end
            start += jobsize
            stop += jobsize
            stop = min(stop,ph.nscenarios)
        end
        # Prepare memory
        log_val = ph.parameters.log
        ph.parameters.log = false
        log!(ph)
        ph.parameters.log = log_val
        # Ensure initialization is finished
        map(wait,active_workers)
        return ph
    end
end

@define_traitfn IsParallel resolve_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A}) where {T <: Real, A <: AbstractVector} = begin
    function resolve_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A},!IsParallel) where {T <: Real, A <: AbstractVector}
        Qs = A(length(ph.subproblems))
        # Update subproblems
        update_primals!(ph.subproblems,ph.ξ)
        # Solve sub problems
        for (i,subproblem) ∈ enumerate(ph.subproblems)
            Qs[i] = subproblem()
        end
        # Return current objective value
        return sum(Qs)
    end

    function resolve_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A},IsParallel) where {T <: Real, A <: AbstractVector}
        active_workers = Vector{Future}(nworkers())
        for w in workers()
            active_workers[w-1] = remotecall(resolve_subproblems!,w,ph.subworkers[w-1],ph.ξ)
        end
        map(wait,active_workers)
        return sum(map(fetch,active_workers))
    end
end

@define_traitfn IsParallel update_iterate!(ph::AbstractProgressiveHedgingSolver) = begin
    function update_iterate!(ph::AbstractProgressiveHedgingSolver,!IsParallel)
        # Update the estimate
        ξ_prev = copy(ph.ξ)
        ph.ξ[:] = sum([subproblem.π*subproblem.x for subproblem in ph.subproblems])
        ξ_diff = norm(ph.ξ-ξ_prev,2)
        # Update dual prices
        update_duals!(ph.subproblems,ph.ξ)
        # Update δ
        ph.solverdata.δ = sqrt(ξ_diff+sum([s.π*norm(s.x-ph.ξ,2) for s in ph.subproblems]))
        return nothing
    end

    function update_iterate!(ph::AbstractProgressiveHedgingSolver,IsParallel)
        active_workers = Vector{Future}(nworkers())
        for w in workers()
            active_workers[w-1] = remotecall(collect_primals,w,ph.subworkers[w-1],length(ph.ξ))
        end
        map(wait,active_workers)
        ξ_prev = copy(ph.ξ)
        ph.ξ[:] = sum(fetch.(active_workers))
        ξ_diff = norm(ph.ξ-ξ_prev,2)
        # Update dual prices
        for w in workers()
            active_workers[w-1] = remotecall(update_duals!,w,ph.subworkers[w-1],ph.ξ)
        end
        map(wait,active_workers)
        ph.solverdata.δ = sqrt(ξ_diff+sum(fetch.(active_workers)))
        return nothing
    end
end

@define_traitfn IsParallel init_workers!(ph::AbstractProgressiveHedgingSolver) = begin
    function init_workers!(ph::AbstractProgressiveHedgingSolver,IsParallel)
        active_workers = Vector{Future}(nworkers())
        for w in workers()
            active_workers[w-1] = remotecall(work_on_subproblems!,
                                             w,
                                             ph.subworkers[w-1],
                                             ph.work[w-1],
                                             ph.cutqueue,
                                             ph.decisions)
        end
        return active_workers
    end
end

@define_traitfn IsParallel close_workers!(ph::AbstractProgressiveHedgingSolver,workers::Vector{Future}) = begin
    function close_workers!(ph::AbstractProgressiveHedgingSolver,workers::Vector{Future},IsParallel)
        @async begin
            close(ph.cutqueue)
            map(wait,workers)
        end
    end
end

@define_traitfn IsParallel fill_submodels!(ph::AbstractProgressiveHedgingSolver,scenarioproblems::StochasticPrograms.ScenarioProblems) = begin
    function fill_submodels!(ph::AbstractProgressiveHedgingSolver,scenarioproblems::StochasticPrograms.ScenarioProblems,!IsParallel)
        for (i,submodel) in enumerate(scenarioproblems.problems)
            fill_submodel!(submodel,ph.subproblems[i])
        end
    end

    function fill_submodels!(ph::AbstractProgressiveHedgingSolver,scenarioproblems::StochasticPrograms.ScenarioProblems,IsParallel)
        j = 0
        for w in workers()
            n = remotecall_fetch((sw)->length(fetch(sw)),w,ph.subworkers[w-1])
            for i = 1:n
                fill_submodel!(scenarioproblems.problems[i+j],remotecall_fetch((sw,i,x)->begin
                                                                               sp = fetch(sw)[i]
                                                                               get_solution(sp)
                                                                               end,
                                                                               w,
                                                                               ph.subworkers[w-1],
                                                                               i,
                                                                               ph.ξ)...)
            end
            j += n
        end
    end
end

@define_traitfn IsParallel fill_submodels!(ph::AbstractProgressiveHedgingSolver,scenarioproblems::StochasticPrograms.DScenarioProblems) = begin
    function fill_submodels!(ph::AbstractProgressiveHedgingSolver,scenarioproblems::StochasticPrograms.DScenarioProblems,!IsParallel)
        active_workers = Vector{Future}(length(scenarioproblems))
        j = 1
        for w in workers()
            n = remotecall_fetch((sp)->length(fetch(sp).problems),w,scenarioproblems[w-1])
            for i in 1:n
                active_workers[j] = remotecall((sp,i,x,μ,λ) -> fill_submodel!(fetch(sp).problems[i],x,μ,λ),
                                               w,
                                               scenarioproblems[w-1],
                                               i,
                                               get_solution(ph.subproblems[j])...)
                j += 1
            end
        end
        @async map(wait,active_workers)
    end

    function fill_submodels!(ph::AbstractProgressiveHedgingSolver,scenarioproblems::StochasticPrograms.DScenarioProblems,IsParallel)
        active_workers = Vector{Future}(nworkers())
        for w in workers()
            active_workers[w-1] = remotecall(fill_submodels!,
                                             w,
                                             ph.subworkers[w-1],
                                             ph.ξ,
                                             scenarioproblems[w-1])
        end
        @async map(wait,active_workers)
    end
end

SubWorker{T,A,S} = RemoteChannel{Channel{Vector{SubProblem{T,A,S}}}}
ScenarioProblems{D,SD,S} = RemoteChannel{Channel{StochasticPrograms.ScenarioProblems{D,SD,S}}}

function load_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A},subsolver::AbstractMathProgSolver) where {T <: Real, A <: AbstractVector}
    for i = 1:ph.nscenarios
        push!(ph.subproblems,SubProblem(WS(ph.structuredmodel,scenario(ph.structuredmodel,i),subsolver),
                                        i,
                                        probability(ph.structuredmodel,i),
                                        ph.parameters.r,
                                        ph.structuredmodel.numCols,
                                        subsolver))
    end
    return ph
end

function load_worker!(scenarioproblems::StochasticPrograms.ScenarioProblems,
                      sp::JuMP.Model,
                      w::Integer,
                      worker::SubWorker,
                      x::AbstractVector,
                      start::Integer,
                      stop::Integer,
                      r::AbstractFloat,
                      subsolver::AbstractMathProgSolver)
    ws_problems = [WS(sp,scenarioproblems.scenariodata[i],subsolver) for i = start:stop]
    πs = [probability(scenarioproblems.scenariodata[i]) for i = start:stop]
    return remotecall(init_subworker!,
                      w,
                      worker,
                      ws_problems,
                      πs,
                      r,
                      sp.numCols,
                      subsolver,
                      collect(start:stop))
end

function load_worker!(scenarioproblems::StochasticPrograms.DScenarioProblems,
                      sp::JuMP.Model,
                      w::Integer,
                      worker::SubWorker,
                      x::AbstractVector,
                      start::Integer,
                      stop::Integer,
                      r::AbstractFloat,
                      subsolver::AbstractMathProgSolver)
    return remotecall(init_subworker!,
                      w,
                      worker,
                      generator(sp,:stage_1),
                      generator(sp,:stage_2),
                      first_stage_data(sp),
                      scenarioproblems[w-1],
                      r,
                      sp.numCols,
                      subsolver,
                      collect(start:stop))
end

function init_subworker!(subworker::SubWorker{T,A,S},
                         submodels::Vector{JuMP.Model},
                         πs::A,
                         r::AbstractFloat,
                         xdim::Integer,
                         subsolver::AbstractMathProgSolver,
                         ids::Vector{Int}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems = Vector{SubProblem{T,A,S}}(length(ids))
    for (i,id) = enumerate(ids)
        subproblems[i] = SubProblem(submodels[i],id,πs[i],r,xdim,subsolver)
    end
    put!(subworker,subproblems)
end

function init_subworker!(subworker::SubWorker{T,A,S},
                         stage_one_generator::Function,
                         stage_two_generator::Function,
                         first_stage::Any,
                         scenarioproblems::ScenarioProblems,
                         r::AbstractFloat,
                         xdim::Integer,
                         subsolver::AbstractMathProgSolver,
                         ids::Vector{Int}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    sp = fetch(scenarioproblems)
    subproblems = Vector{SubProblem{T,A,S}}(length(ids))
    for (i,id) = enumerate(ids)
        subproblems[i] = SubProblem(_WS(stage_one_generator,stage_two_generator,first_stage,stage_data(sp),scenario(sp,i),subsolver),
                                    id,
                                    probability(scenario(sp,i)),
                                    r,
                                    xdim,
                                    subsolver)
    end
    put!(subworker,subproblems)
end

function resolve_subproblems!(subworker::SubWorker{T,A,S},ξ::AbstractVector) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    Qs = A(length(subproblems))
    for (i,subproblem) ∈ enumerate(subproblems)
        update_primal!(subproblem,ξ)
        Qs[i] = subproblem()
    end
    return sum(Qs)
end

function collect_primals(subworker::SubWorker{T,A,S},n::Integer) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    if length(subproblems) > 0
        return sum([subproblem.π*subproblem.x for subproblem in subproblems])
    else
        return zeros(T,n)
    end
end

function update_duals!(subworker::SubWorker{T,A,S},ξ::AbstractVector) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    if length(subproblems) > 0
        update_duals!(subproblems,ξ)
        return sum([s.π*norm(s.x-ξ,2) for s in subproblems])
    else
        return zero(T)
    end
end

function fill_submodels!(subworker::SubWorker{T,A,S},
                         x::A,
                         scenarioproblems::ScenarioProblems) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    sp = fetch(scenarioproblems)
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    for (i,submodel) in enumerate(sp.problems)
        fill_submodel!(submodel,subproblems[i])
    end
end

function fill_submodel!(submodel::JuMP.Model,subproblem::SubProblem)
    fill_submodel!(submodel,get_solution(subproblem)...)
end

function fill_submodel!(submodel::JuMP.Model,x::AbstractVector,μ::AbstractVector,λ::AbstractVector)
    submodel.colVal = x
    submodel.redCosts = μ
    submodel.linconstrDuals = λ
    submodel.objVal = JuMP.prepAffObjective(submodel)⋅x
end
