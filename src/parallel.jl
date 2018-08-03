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

    # function init_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A,M,S},subsolver::AbstractMathProgSolver,IsParallel) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
    #     @unpack κ = ph.parameters
    #     # Partitioning
    #     (jobsize,extra) = divrem(ph.nscenarios,nworkers())
    #     # One extra to guarantee coverage
    #     if extra > 0
    #         jobsize += 1
    #     end
    #     # Load initial decision
    #     put!(ph.decisions,1,ph.x)
    #     # Create subproblems on worker processes
    #     m = ph.structuredmodel
    #     start = 1
    #     stop = jobsize
    #     active_workers = Vector{Future}(nworkers())
    #     for w in workers()
    #         ph.work[w-1] = RemoteChannel(() -> Channel{Int}(round(Int,10/κ)), w)
    #         put!(ph.work[w-1],1)
    #         ph.subworkers[w-1] = RemoteChannel(() -> Channel{Vector{SubProblem{T,A,S}}}(1), w)
    #         active_workers[w-1] = load_worker!(scenarioproblems(m),w,ph.subworkers[w-1],ph.x,start,stop,subsolver)
    #         if start > ph.nscenarios
    #             continue
    #         end
    #         start += jobsize
    #         stop += jobsize
    #         stop = min(stop,ph.nscenarios)
    #     end
    #     # Prepare memory
    #     push!(ph.subobjectives,zeros(ph.nscenarios))
    #     push!(ph.finished,0)
    #     log_val = ph.parameters.log
    #     ph.parameters.log = false
    #     log!(ph)
    #     ph.parameters.log = log_val
    #     # Ensure initialization is finished
    #     map(wait,active_workers)
    #     return ph
    # end
end

@define_traitfn IsParallel iterate!(ph::AbstractProgressiveHedgingSolver) = begin
    function iterate!(ph::AbstractProgressiveHedgingSolver,!IsParallel)
        iterate_nominal!(ph)
    end

    function iterate!(ph::AbstractProgressiveHedgingSolver,IsParallel)
        iterate_parallel!(ph)
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

    # function fill_submodels!(ph::AbstractProgressiveHedgingSolver,scenarioproblems::StochasticPrograms.ScenarioProblems,IsParallel)
    #     j = 0
    #     for w in workers()
    #         n = remotecall_fetch((sw)->length(fetch(sw)),w,ph.subworkers[w-1])
    #         for i = 1:n
    #             fill_submodel!(scenarioproblems.problems[i+j],remotecall_fetch((sw,i,x)->begin
    #                                                                            sp = fetch(sw)[i]
    #                                                                            sp(x)
    #                                                                            get_solution(sp)
    #                                                                            end,
    #                                                                            w,
    #                                                                            ph.subworkers[w-1],
    #                                                                            i,
    #                                                                            ph.x)...)
    #         end
    #         j += n
    #     end
    # end
end

@define_traitfn IsParallel fill_submodels!(ph::AbstractProgressiveHedgingSolver,scenarioproblems::StochasticPrograms.DScenarioProblems) = begin
    function fill_submodels!(ph::AbstractProgressiveHedgingSolver,scenarioproblems::StochasticPrograms.DScenarioProblems,!IsParallel)
        active_workers = Vector{Future}(nworkers())
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

    # function fill_submodels!(ph::AbstractProgressiveHedgingSolver,scenarioproblems::StochasticPrograms.DScenarioProblems,IsParallel)
    #     active_workers = Vector{Future}(nworkers())
    #     for w in workers()
    #         active_workers[w-1] = remotecall(fill_submodels!,
    #                                          w,
    #                                          ph.subworkers[w-1],
    #                                          ph.x,
    #                                          scenarioproblems[w-1])
    #     end
    #     @async map(wait,active_workers)
    # end
end

# Parallel routines #
# ======================================================================== #
# mutable struct DecisionChannel{A <: AbstractArray} <: AbstractChannel
#     decisions::Dict{Int,A}
#     cond_take::Condition
#     DecisionChannel(decisions::Dict{Int,A}) where A <: AbstractArray = new{A}(decisions, Condition())
# end

# function put!(channel::DecisionChannel, t, x)
#     channel.decisions[t] = copy(x)
#     notify(channel.cond_take)
#     return channel
# end

# function take!(channel::DecisionChannel, t)
#     x = fetch(channel,t)
#     delete!(channel.decisions, t)
#     return x
# end

# isready(channel::DecisionChannel) = length(channel.decisions) > 1
# isready(channel::DecisionChannel, t) = haskey(channel.decisions,t)

# function fetch(channel::DecisionChannel, t)
#     wait(channel,t)
#     return channel.decisions[t]
# end

# function wait(channel::DecisionChannel, t)
#     while !isready(channel, t)
#         wait(channel.cond_take)
#     end
# end

# SubWorker{T,A,S} = RemoteChannel{Channel{Vector{SubProblem{T,A,S}}}}
# ScenarioProblems{D,SD,S} = RemoteChannel{Channel{StochasticPrograms.ScenarioProblems{D,SD,S}}}
# Work = RemoteChannel{Channel{Int}}
# Decisions{A} = RemoteChannel{DecisionChannel{A}}
# QCut{T} = Tuple{Int,T,SparseHyperPlane{T}}
# CutQueue{T} = RemoteChannel{Channel{QCut{T}}}

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

# function load_worker!(sp::StochasticPrograms.ScenarioProblems,
#                       w::Integer,
#                       worker::SubWorker,
#                       x::AbstractVector,
#                       start::Integer,
#                       stop::Integer,
#                       subsolver::AbstractMathProgSolver)
#     problems = [sp.problems[i] for i = start:stop]
#     πs = [probability(sp.scenariodata[i]) for i = start:stop]
#     return remotecall(init_subworker!,
#                       w,
#                       worker,
#                       sp.parent,
#                       problems,
#                       πs,
#                       x,
#                       subsolver,
#                       collect(start:stop))
# end

# function load_worker!(sp::StochasticPrograms.DScenarioProblems,
#                       w::Integer,
#                       worker::SubWorker,
#                       x::AbstractVector,
#                       start::Integer,
#                       stop::Integer,
#                       subsolver::AbstractMathProgSolver)
#     return remotecall(init_subworker!,
#                       w,
#                       worker,
#                       sp[w-1],
#                       x,
#                       subsolver,
#                       collect(start:stop))
# end

# function init_subworker!(subworker::SubWorker{T,A,S},
#                          parent::JuMP.Model,
#                          submodels::Vector{JuMP.Model},
#                          πs::A,
#                          x::A,
#                          subsolver::AbstractMathProgSolver,
#                          ids::Vector{Int}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
#     subproblems = Vector{SubProblem{T,A,S}}(length(ids))
#     for (i,id) = enumerate(ids)
#         y₀ = convert(A,rand(submodels[i].numCols))
#         subproblems[i] = SubProblem(submodels[i],parent,id,πs[i],x,y₀,subsolver)
#     end
#     put!(subworker,subproblems)
# end

# function init_subworker!(subworker::SubWorker{T,A,S},
#                          scenarioproblems::ScenarioProblems,
#                          x::A,
#                          subsolver::AbstractMathProgSolver,
#                          ids::Vector{Int}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
#     sp = fetch(scenarioproblems)
#     subproblems = Vector{SubProblem{T,A,S}}(length(ids))
#     for (i,id) = enumerate(ids)
#         y₀ = convert(A,rand(sp.problems[i].numCols))
#         subproblems[i] = SubProblem(sp.problems[i],sp.parent,id,probability(sp.scenariodata[i]),x,y₀,subsolver)
#     end
#     put!(subworker,subproblems)
# end

# function work_on_subproblems!(subworker::SubWorker{T,A,S},
#                               work::Work,
#                               cuts::CutQueue{T},
#                               decisions::Decisions{A}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
#     subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
#     while true
#         wait(work)
#         t::Int = take!(work)
#         if t == -1
#             # Worker finished
#             return
#         end
#         x::A = fetch(decisions,t)
#         for subproblem in subproblems
#             @schedule begin
#                 update_subproblem!(subproblem,x)
#                 cut = subproblem()
#                 Q::T = cut(x)
#                 try
#                     put!(cuts,(t,Q,cut))
#                 catch err
#                     if err isa InvalidStateException
#                         # Master closed the cut channel. Worker finished
#                         return
#                     end
#                 end
#             end
#         end
#     end
# end

# function fill_submodels!(subworker::SubWorker{T,A,S},
#                          x::A,
#                          scenarioproblems::ScenarioProblems) where {T <: Real, A <: AbstractArray, S <: LQSolver}
#     sp = fetch(scenarioproblems)
#     subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
#     for (i,submodel) in enumerate(sp.problems)
#         subproblems[i](x)
#         fill_submodel!(submodel,subproblems[i])
#     end
# end

function fill_submodel!(submodel::JuMP.Model,subproblem::SubProblem)
    fill_submodel!(submodel,get_solution(subproblem)...)
end

function fill_submodel!(submodel::JuMP.Model,x::AbstractVector,μ::AbstractVector,λ::AbstractVector)
    submodel.colVal = x
    submodel.redCosts = μ
    submodel.linconstrDuals = λ
    submodel.objVal = JuMP.prepAffObjective(model)⋅x
end

# function iterate_parallel!(ph::AbstractProgressiveHedgingSolver{T,A,M,S}) where {T <: Real, A <: AbstractVector, M <: LQSolver, S <: LQSolver}
#     wait(ph.cutqueue)
#     while isready(ph.cutqueue)
#         # Add new cuts from subworkers
#         t::Int,Q::T,cut::SparseHyperPlane{T} = take!(ph.cutqueue)
#         if !bounded(cut)
#             map(w->put!(w,-1),ph.work)
#             warn("Subproblem ",cut.id," is unbounded, aborting procedure.")
#             return :Unbounded
#         end
#         addcut!(ph,cut,Q)
#         ph.subobjectives[t][cut.id] = Q
#         ph.finished[t] += 1
#         if ph.finished[t] == ph.nscenarios
#             ph.solverdata.timestamp = t
#             ph.x[:] = fetch(ph.decisions,t)
#             ph.Q_history[t] = current_objective_value(ph,ph.subobjectives[t])
#             ph.solverdata.Q = ph.Q_history[t]
#             ph.solverdata.θ = t > 1 ? ph.θ_history[t-1] : -1e10
#             take_step!(ph)
#             ph.solverdata.θ = ph.θ_history[t]
#             # Check if optimal
#             if check_optimality(ph)
#                 # Optimal, tell workers to stop
#                 map(w->put!(w,t),ph.work)
#                 map(w->put!(w,-1),ph.work)
#                 # Final log
#                 log!(ph,ph.solverdata.iterations)
#                 return :Optimal
#             end
#         end
#     end
#     # Resolve master
#     t = ph.solverdata.iterations
#     if ph.finished[t] >= ph.parameters.κ*ph.nscenarios && length(ph.cuts) >= ph.nscenarios
#         try
#             solve_problem!(ph,ph.mastersolver)
#         catch
#             # Master problem could not be solved for some reason.
#             @unpack Q,θ = ph.solverdata
#             gap = abs(θ-Q)/(abs(Q)+1e-10)
#             warn("Master problem could not be solved, solver returned status $(status(ph.mastersolver)). The following relative tolerance was reached: $(@sprintf("%.1e",gap)). Aborting procedure.")
#             map(w->put!(w,-1),ph.work)
#             return :StoppedPrematurely
#         end
#         if status(ph.mastersolver) == :Infeasible
#             warn("Master is infeasible. Aborting procedure.")
#             map(w->put!(w,-1),ph.work)
#             return :Infeasible
#         end
#         # Update master solution
#         update_solution!(ph)
#         θ = calculate_estimate(ph)
#         if t > 1 && abs(θ-ph.θ_history[t-1]) <= 10*ph.parameters.τ*abs(1e-10+θ) && ph.finished[t] != ph.nscenarios
#             # Not enough new information in master. Repeat iterate
#             return :Valid
#         end
#         ph.solverdata.θ = θ
#         ph.θ_history[t] = ph.solverdata.θ
#         # Project (if applicable)
#         project!(ph)
#         # If all work is finished at this timestamp, check optimality
#         if ph.finished[t] == ph.nscenarios
#             # Check if optimal
#             if check_optimality(ph)
#                 # Optimal, tell workers to stop
#                 map(w->put!(w,t),ph.work)
#                 map(w->put!(w,-1),ph.work)
#                 # Final log
#                 log!(ph,t)
#                 return :Optimal
#             end
#         end
#         # Log progress at current timestamp
#         log_regularization!(ph,t)
#         # Send new decision vector to workers
#         put!(ph.decisions,t+1,ph.x)
#         for w in ph.work
#             put!(w,t+1)
#         end
#         # Prepare memory for next iteration
#         push!(ph.subobjectives,zeros(ph.nscenarios))
#         push!(ph.finished,0)
#         # Log progress
#         log!(ph)
#         ph.θ_history[t+1] = -Inf
#     end
#     # Just return a valid status for this iteration
#     return :Valid
# end
