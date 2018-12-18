# ------------------------------------------------------------
# Parallel -> Algorithm is run in parallel
# ------------------------------------------------------------
@define_trait Parallel

@define_traitfn Parallel init_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A,S}, subsolver::MPB.AbstractMathProgSolver) where {T <: Real, A <: AbstractVector, S <: LQSolver} = begin
    function init_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A,S}, subsolver::MPB.AbstractMathProgSolver, !Parallel) where {T <: Real, A <: AbstractVector, S <: LQSolver}
        # Prepare the subproblems
        m = ph.stochasticprogram
        load_subproblems!(ph, subsolver)
        ph.ξ[:] = sum([subproblem.π*subproblem.x for subproblem in ph.subproblems])
        return ph
    end

    function init_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A,S}, subsolver::MPB.AbstractMathProgSolver, Parallel) where {T <: Real, A <: AbstractVector, S <: LQSolver}
        # Partitioning
        (jobsize, extra) = divrem(ph.nscenarios, nworkers())
        # One extra to guarantee coverage
        if extra > 0
            jobsize += 1
        end
        # Create subproblems on worker processes
        m = ph.stochasticprogram
        start = 1
        stop = jobsize
        active_workers = Vector{Future}(undef, nworkers())
        for w in workers()
            ph.subworkers[w-1] = RemoteChannel(() -> Channel{Vector{SubProblem{T,A,S}}}(1), w)
            active_workers[w-1] = load_worker!(scenarioproblems(m), m, w, ph.subworkers[w-1], ph.ξ, start, stop, subsolver)
            if start > ph.nscenarios
                continue
            end
            start += jobsize
            stop += jobsize
            stop = min(stop, ph.nscenarios)
        end
        # Prepare memory
        log_val = ph.parameters.log
        ph.parameters.log = false
        log!(ph)
        ph.parameters.log = log_val
        # Ensure initialization is finished
        map(wait, active_workers)
        return ph
    end
end

@define_traitfn Parallel resolve_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A}) where {T <: Real, A <: AbstractVector} = begin
    function resolve_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A}, !Parallel) where {T <: Real, A <: AbstractVector}
        Qs = A(undef, length(ph.subproblems))
        # Update subproblems
        reformulate_subproblems!(ph.subproblems, ph.ξ, penalty(ph))
        # Solve sub problems
        for (i, subproblem) ∈ enumerate(ph.subproblems)
            Qs[i] = subproblem()
        end
        # Return current objective value
        return sum(Qs)
    end

    function resolve_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A}, Parallel) where {T <: Real, A <: AbstractVector}
        active_workers = Vector{Future}(undef, nworkers())
        for w in workers()
            active_workers[w-1] = remotecall(resolve_subproblems!, w, ph.subworkers[w-1], ph.ξ, penalty(ph))
        end
        map(wait, active_workers)
        return sum(map(fetch, active_workers))
    end
end

@define_traitfn Parallel update_iterate!(ph::AbstractProgressiveHedgingSolver) = begin
    function update_iterate!(ph::AbstractProgressiveHedgingSolver, !Parallel)
        # Update the estimate
        ξ_prev = copy(ph.ξ)
        ph.ξ[:] = sum([subproblem.π*subproblem.x for subproblem in ph.subproblems])
        # Update δ₁
        ph.solverdata.δ₁ = norm(ph.ξ-ξ_prev, 2)^2
        return nothing
    end

    function update_iterate!(ph::AbstractProgressiveHedgingSolver, Parallel)
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
end

@define_traitfn Parallel update_subproblems!(ph::AbstractProgressiveHedgingSolver) = begin
    function update_subproblems!(ph::AbstractProgressiveHedgingSolver, !Parallel)
        # Update dual prices
        update_subproblems!(ph.subproblems, ph.ξ, penalty(ph))
        return nothing
    end

    function update_subproblems!(ph::AbstractProgressiveHedgingSolver, Parallel)
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
end

@define_traitfn Parallel update_dual_gap!(ph::AbstractProgressiveHedgingSolver) = begin
    function update_dual_gap!(ph::AbstractProgressiveHedgingSolver, !Parallel)
        # Update δ₂
        ph.solverdata.δ₂ = sum([s.π*norm(s.x-ph.ξ,2)^2 for s in ph.subproblems])
        return nothing
    end

    function update_dual_gap!(ph::AbstractProgressiveHedgingSolver, Parallel)
        active_workers = Vector{Future}(undef, nworkers())
        # Update δ₂
        for w in workers()
            active_workers[w-1] = remotecall((sw,ξ)->begin
                                                 subproblems = fetch(sw)
                                                 if length(subproblems) > 0
                                                     return sum([s.π*norm(s.x-ξ,2)^2 for s in subproblems])
                                                 else
                                                     return zero(eltype(ξ))
                                                 end
                                             end,
                                             w,
                                             ph.subworkers[w-1],
                                             ph.ξ)
        end
        map(wait, active_workers)
        ph.solverdata.δ₂ = sum(fetch.(active_workers))
        return nothing
    end
end

@define_traitfn Parallel calculate_objective_value(ph::AbstractProgressiveHedgingSolver) = begin
    function calculate_objective_value(ph::AbstractProgressiveHedgingSolver, !Parallel)
        return sum([get_objective_value(s) for s in ph.subproblems])
    end

    function calculate_objective_value(ph::AbstractProgressiveHedgingSolver, Parallel)
        active_workers = Vector{Future}(undef, nworkers())
        # Update δ₂
        for w in workers()
            active_workers[w-1] = remotecall((sw)->begin
                                                 subproblems = fetch(sw)
                                                 if length(subproblems) > 0
                                                     return sum([get_objective_value(s) for s in subproblems])
                                                 else
                                                     return 0.0
                                                 end
                                             end,
                                             w,
                                             ph.subworkers[w-1])
        end
        map(wait, active_workers)
        return sum(fetch.(active_workers))
    end
end

@define_traitfn Parallel init_workers!(ph::AbstractProgressiveHedgingSolver) = begin
    function init_workers!(ph::AbstractProgressiveHedgingSolver, Parallel)
        active_workers = Vector{Future}(undef, nworkers())
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

@define_traitfn Parallel close_workers!(ph::AbstractProgressiveHedgingSolver, workers::Vector{Future}) = begin
    function close_workers!(ph::AbstractProgressiveHedgingSolver, workers::Vector{Future}, Parallel)
        @async begin
            close(ph.cutqueue)
            map(wait, workers)
        end
    end
end

@define_traitfn Parallel fill_submodels!(ph::AbstractProgressiveHedgingSolver, scenarioproblems::StochasticPrograms.ScenarioProblems) = begin
    function fill_submodels!(ph::AbstractProgressiveHedgingSolver, scenarioproblems::StochasticPrograms.ScenarioProblems, !Parallel)
        for (i, submodel) in enumerate(scenarioproblems.problems)
            fill_submodel!(submodel, ph.subproblems[i])
        end
    end

    function fill_submodels!(ph::AbstractProgressiveHedgingSolver, scenarioproblems::StochasticPrograms.ScenarioProblems, Parallel)
        j = 0
        for w in workers()
            n = remotecall_fetch((sw)->length(fetch(sw)), w, ph.subworkers[w-1])
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

@define_traitfn Parallel fill_submodels!(ph::AbstractProgressiveHedgingSolver, scenarioproblems::StochasticPrograms.DScenarioProblems) = begin
    function fill_submodels!(ph::AbstractProgressiveHedgingSolver, scenarioproblems::StochasticPrograms.DScenarioProblems, !Parallel)
        active_workers = Vector{Future}(undef, nsubproblems(scenarioproblems))
        j = 1
        for w in workers()
            n = remotecall_fetch((sp)->length(fetch(sp).problems), w, scenarioproblems[w-1])
            for i in 1:n
                active_workers[j] = remotecall((sp,i,x,μ,λ) -> fill_submodel!(fetch(sp).problems[i],x,μ,λ),
                                               w,
                                               scenarioproblems[w-1],
                                               i,
                                               get_solution(ph.subproblems[j])...)
                j += 1
            end
        end
        map(wait, active_workers)
    end

    function fill_submodels!(ph::AbstractProgressiveHedgingSolver, scenarioproblems::StochasticPrograms.DScenarioProblems, Parallel)
        active_workers = Vector{Future}(undef, nworkers())
        for w in workers()
            active_workers[w-1] = remotecall(fill_submodels!,
                                             w,
                                             ph.subworkers[w-1],
                                             ph.ξ,
                                             scenarioproblems[w-1])
        end
        map(wait, active_workers)
    end
end

SubWorker{T,A,S} = RemoteChannel{Channel{Vector{SubProblem{T,A,S}}}}
ScenarioProblems{D,S} = RemoteChannel{Channel{StochasticPrograms.ScenarioProblems{D,S}}}

function load_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A}, subsolver::MPB.AbstractMathProgSolver) where {T <: Real, A <: AbstractVector}
    for i = 1:ph.nscenarios
        push!(ph.subproblems,SubProblem(WS(ph.stochasticprogram, scenario(ph.stochasticprogram,i); solver = subsolver),
                                        i,
                                        probability(ph.stochasticprogram,i),
                                        decision_length(ph.stochasticprogram),
                                        subsolver))
    end
    return ph
end

function load_worker!(scenarioproblems::StochasticPrograms.ScenarioProblems,
                      sp::StochasticProgram,
                      w::Integer,
                      worker::SubWorker,
                      x::AbstractVector,
                      start::Integer,
                      stop::Integer,
                      subsolver::MPB.AbstractMathProgSolver)
    ws_problems = [WS(sp, scenarioproblems.scenarios[i], solver = subsolver) for i = start:stop]
    πs = [probability(scenarioproblems.scenarios[i]) for i = start:stop]
    return remotecall(init_subworker!,
                      w,
                      worker,
                      ws_problems,
                      πs,
                      decision_length(sp),
                      subsolver,
                      collect(start:stop))
end

function load_worker!(scenarioproblems::StochasticPrograms.DScenarioProblems,
                      sp::StochasticProgram,
                      w::Integer,
                      worker::SubWorker,
                      x::AbstractVector,
                      start::Integer,
                      stop::Integer,
                      subsolver::MPB.AbstractMathProgSolver)
    return remotecall(init_subworker!,
                      w,
                      worker,
                      generator(sp,:stage_1),
                      generator(sp,:stage_2),
                      first_stage_data(sp),
                      scenarioproblems[w-1],
                      decision_length(sp),
                      subsolver,
                      collect(start:stop))
end

function init_subworker!(subworker::SubWorker{T,A,S},
                         submodels::Vector{JuMP.Model},
                         πs::A,
                         xdim::Integer,
                         subsolver::MPB.AbstractMathProgSolver,
                         ids::Vector{Int}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems = Vector{SubProblem{T,A,S}}(undef, length(ids))
    for (i,id) = enumerate(ids)
        subproblems[i] = SubProblem(submodels[i], id, πs[i], xdim, subsolver)
    end
    put!(subworker, subproblems)
end

function init_subworker!(subworker::SubWorker{T,A,S},
                         stage_one_generator::Function,
                         stage_two_generator::Function,
                         first_stage::Any,
                         scenarioproblems::ScenarioProblems,
                         xdim::Integer,
                         subsolver::MPB.AbstractMathProgSolver,
                         ids::Vector{Int}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    sp = fetch(scenarioproblems)
    subproblems = Vector{SubProblem{T,A,S}}(undef, length(ids))
    for (i,id) = enumerate(ids)
        subproblems[i] = SubProblem(_WS(stage_one_generator, stage_two_generator, first_stage, stage_data(sp), scenario(sp,i), subsolver),
                                    id,
                                    probability(scenario(sp,i)),
                                    xdim,
                                    subsolver)
    end
    put!(subworker, subproblems)
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

function fill_submodels!(subworker::SubWorker{T,A,S},
                         x::A,
                         scenarioproblems::ScenarioProblems) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    sp = fetch(scenarioproblems)
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    for (i, submodel) in enumerate(sp.problems)
        fill_submodel!(submodel, subproblems[i])
    end
end

function fill_submodel!(submodel::JuMP.Model, subproblem::SubProblem)
    fill_submodel!(submodel, get_solution(subproblem)...)
end

function fill_submodel!(submodel::JuMP.Model, x::AbstractVector, μ::AbstractVector, λ::AbstractVector)
    submodel.colVal = x
    submodel.redCosts = μ
    submodel.linconstrDuals = λ
    submodel.objVal = JuMP.prepAffObjective(submodel)⋅x
end
