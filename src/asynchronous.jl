@implement_traitfn function init_subproblems!(ph::AbstractProgressiveHedgingSolver{T,A,S}, subsolver::MPB.AbstractMathProgSolver, Asynchronous) where {T <: Real, A <: AbstractVector, S <: LQSolver}
    @unpack κ = ph.parameters
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
    # Ensure initialization is finished
    map(wait, active_workers)
    # Continue preparation
    for w in workers()
        ph.work[w-1] = RemoteChannel(() -> Channel{Int}(round(Int,10/κ)), w)
        ph.x̄[w-1] = remotecall_fetch((sw, xdim)->begin
                                         subproblems = fetch(sw)
                                         if length(subproblems) > 0
                                             x̄ = sum([s.π*s.x for s in subproblems])
                                             return RemoteChannel(()->RunningAverageChannel(x̄, [s.x for s in subproblems]), myid())
                                         else
                                             return RemoteChannel(()->RunningAverageChannel(zeros(T,xdim), Vector{A}()), myid())
                                         end
                                     end, w, ph.subworkers[w-1], decision_length(m))
        ph.ȳ[w-1] = remotecall_fetch((sw, xdim)->begin
                                     subproblems = fetch(sw)
                                         if length(subproblems) > 0
                                             return RemoteChannel(()->RunningAverageChannel(zeros(T,xdim), [zero(s.ρ) for s in subproblems]), myid())
                                         else
                                             return RemoteChannel(()->RunningAverageChannel(zeros(T,xdim), Vector{A}()), myid())
                                         end
                                     end, w, ph.subworkers[w-1], decision_length(m))
        put!(ph.work[w-1], 1)
    end
    # Prepare memory
    push!(ph.subobjectives, zeros(nscenarios(ph)))
    push!(ph.finished, 0)
    log_val = ph.parameters.log
    ph.parameters.log = false
    log!(ph)
    ph.parameters.log = log_val
    update_iterate!(ph)
    return ph
end

@implement_traitfn function update_iterate!(ph::AbstractProgressiveHedgingSolver, Asynchronous)
    ξ_prev = copy(ph.ξ)
    ph.ξ[:] = sum([fetch(x̄) for x̄ in ph.x̄])
    # Update δ₁
    ph.solverdata.δ₁ = norm(ph.ξ-ξ_prev, 2)^2
end

@define_traitfn Parallel init_workers!(ph::AbstractProgressiveHedgingSolver) = begin
    function init_workers!(ph::AbstractProgressiveHedgingSolver, Asynchronous)
        # Load initial decision
        put!(ph.decisions, 1, ph.ξ)
        put!(ph.r, 1, penalty(ph))
        for w in workers()
            ph.active_workers[w-1] = remotecall(work_on_subproblems!,
                                                w,
                                                ph.subworkers[w-1],
                                                ph.work[w-1],
                                                ph.progressqueue,
                                                ph.x̄[w-1],
                                                ph.ȳ[w-1],
                                                ph.decisions,
                                                ph.u,
                                                ph.r,
                                                ph.θ)
        end
        return nothing
    end
end

@define_traitfn Parallel close_workers!(ph::AbstractProgressiveHedgingSolver) = begin
    function close_workers!(ph::AbstractProgressiveHedgingSolver, Asynchronous)
        map(wait, ph.active_workers)
    end
end

mutable struct IterationChannel{D} <: AbstractChannel{D}
    data::Dict{Int,D}
    cond_take::Condition
    IterationChannel(data::Dict{Int,D}) where D = new{D}(data, Condition())
end

function put!(channel::IterationChannel, t, x)
    channel.data[t] = copy(x)
    notify(channel.cond_take)
    return channel
end

function take!(channel::IterationChannel, t)
    x = fetch(channel, t)
    delete!(channel.data, t)
    return x
end

isready(channel::IterationChannel) = length(channel.data) > 1
isready(channel::IterationChannel, t) = haskey(channel.data, t)

function fetch(channel::IterationChannel, t)
    wait(channel, t)
    return channel.data[t]
end

function wait(channel::IterationChannel, t)
    while !isready(channel, t)
        wait(channel.cond_take)
    end
end

mutable struct RunningAverageChannel{A <: AbstractArray} <: AbstractChannel{A}
    average::A
    data::Vector{A}
    buffer::Dict{Int,A}
    cond_put::Condition
    RunningAverageChannel(average::A, data::Vector{A}) where A <: AbstractArray = new{A}(average, data, Dict{Int,A}(), Condition())
end

function take!(channel::RunningAverageChannel, i::Integer)
    channel.buffer[i] = copy(channel.data[i])
end

function put!(channel::RunningAverageChannel, i::Integer, π::AbstractFloat)
    channel.average -= π*channel.buffer[i]
    channel.average += π*channel.data[i]
    delete!(channel.buffer, i)
    notify(channel.cond_put)
    return channel
end

function put!(channel::RunningAverageChannel, i::Integer, x::AbstractArray, π::AbstractFloat)
    channel.average -= π*channel.buffer[i]
    channel.average += π*x
    channel.data[i] = copy(x)
    delete!(channel.buffer, i)
    notify(channel.cond_put)
    return channel
end

isready(channel::RunningAverageChannel) = length(channel.buffer) == 0

function fetch(channel::RunningAverageChannel)
    wait(channel)
    return channel.average
end

function fetch(channel::RunningAverageChannel, i::Integer)
    return channel.data[i]
end

function wait(channel::RunningAverageChannel)
    while !isready(channel)
        wait(channel.cond_put)
    end
end

Work = RemoteChannel{Channel{Int}}
IteratedValue{T <: AbstractFloat} = RemoteChannel{IterationChannel{T}}
RunningAverage{A <: AbstractArray} = RemoteChannel{RunningAverageChannel{A}}
Decisions{A <: AbstractArray} = RemoteChannel{IterationChannel{A}}
Progress{T <: AbstractFloat} = Tuple{Int,Int,T}
ProgressQueue{T <: AbstractFloat} = RemoteChannel{Channel{Progress{T}}}

function work_on_subproblems!(subworker::SubWorker{T,A,S},
                              work::Work,
                              progress::ProgressQueue{T},
                              x̄::RunningAverage{A},
                              ȳ::RunningAverage{A},
                              decisions::Decisions{A},
                              ξ̄::Decisions{A},
                              r::IteratedValue{T},
                              θ::IteratedValue{T}) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    if isempty(subproblems)
       # Workers has nothing do to, return.
       return
    end
    while true
        t::Int = try
            wait(work)
            take!(work)
        catch err
            if err isa InvalidStateException
                # Master closed the work channel. Worker finished
                return
            end
        end
        if t == -1
            # Worker finished
            return
        end
        ξ::A = fetch(decisions, t)
        if t > 1
            update_subproblems!(subproblems, ξ, fetch(r,t))
        end
        @sync for (i,subproblem) ∈ enumerate(subproblems)
            @async begin
                take!(x̄, i)
                take!(ȳ, i)
                reformulate_subproblem!(subproblem, ξ, fetch(r,t))
                Q::T = subproblem()
                put!(x̄, i, subproblem.π)
                put!(ȳ, i, subproblem.ρ + fetch(r,t)*(subproblem.x - ξ), subproblem.π)
                put!(progress, (t,i,Q))
            end
        end
    end
end

function calculate_tau!(ph::AbstractProgressiveHedgingSolver{T}) where T <: Real
    active_workers = Vector{Future}(undef, nworkers())
    for w in workers()
        active_workers[w-1] = remotecall(collect_primals, w, ph.subworkers[w-1], length(ph.ξ))
    end
end

function calculate_theta!(ph::AbstractProgressiveHedgingSolver{T}, t::Integer) where T <: Real
    @unpack τ = ph.solverdata
    @unpack ν = ph.parameters
    active_workers = Vector{Future}(undef, nworkers())
    for w in workers()
        active_workers[w-1] = remotecall(calculate_theta_part, w, ph.subworkers[w-1], ph.ȳ[w-1], fetch(ph.decisions, t))
    end
    map(wait, active_workers)
    # Update θ
    ph.solverdata.θ = (ν/τ)*max(0, sum(fetch.(active_workers)))
end

function calculate_theta_part(subworker::SubWorker{T,A,S}, ȳ::RunningAverage{A}, ξ::AbstractVector) where {T <: Real, A <: AbstractArray, S <: LQSolver}
    subproblems::Vector{SubProblem{T,A,S}} = fetch(subworker)
    if length(subproblems) > 0
        return sum([subproblem.π*(ξ-subproblem.x)⋅(subproblem.ρ-fetch(ȳ)) for subproblem in subproblems])
    else
        return zero(T)
    end
end

function iterate_async!(ph::AbstractProgressiveHedgingSolver{T}) where T <: Real
    wait(ph.progressqueue)
    while isready(ph.progressqueue)
        # Add new cuts from subworkers
        t::Int, i::Int, Q::T = take!(ph.progressqueue)
        if Q == Inf
            @warn "Subproblem $(i) is infeasible, aborting procedure."
            return :Infeasible
        end
        ph.subobjectives[t][i] = Q
        ph.finished[t] += 1
        if ph.finished[t] == nscenarios(ph)
            ph.solverdata.timestamp = t
            # Update objective
            ph.Q_history[t] = current_objective_value(ph, ph.subobjectives[t])
            ph.solverdata.Q = ph.Q_history[t]
            # Update iterate
            update_iterate!(ph)
            # Get dual gap
            update_dual_gap!(ph)
            ph.dual_gaps[t] = ph.solverdata.δ₂
            # Update penalty (if applicable)
            update_penalty!(ph)
            # Update progress
            @unpack δ₁, δ₂ = ph.solverdata
            ph.solverdata.δ = sqrt(δ₁ + δ₂)/(1e-10+norm(ph.ξ,2))
            # Check if optimal
            if check_optimality(ph)
                # Optimal, tell workers to stop
                map((w,aw)->!isready(aw) && put!(w,t), ph.work, ph.active_workers)
                map((w,aw)->!isready(aw) && put!(w,-1), ph.work, ph.active_workers)
                # Final log
                log!(ph)
                return :Optimal
            end
        end
    end
    # Project and generate new iterate
    t = ph.solverdata.iterations
    if ph.finished[t] >= ph.parameters.κ*nscenarios(ph)
        # ph.v[:] = sum([fetch(ȳ) for ȳ in ph.ȳ])
        # # Get dual gap
        # update_dual_gap!(ph)
        # # Calculate τ
        # τ = ph.solverdata.δ₂ + norm(ph.v,2)^2
        # if τ <= ph.parameters.τ
        #     # Projections are zero, terminate
        #     ph.ξ[:] = fetch(ph.decisions, t)
        #     return :Optimal
        # end
        # @pack! ph.solverdata = τ
        # # Calculate θ
        # calculate_theta!(ph, t)
        @unpack θ = ph.solverdata
        # Update iterate
        update_iterate!(ph)
        # Send new work to workers
        # put!(ph.decisions, t+1, fetch(ph.decisions, t) + θ*ph.v)
        put!(ph.decisions, t+1, ph.ξ)
        put!(ph.u, t+1, sum([fetch(x̄) for x̄ in ph.x̄]))
        put!(ph.r, t+1, penalty(ph))
        put!(ph.θ, t+1, θ)
        map((w,aw)->!isready(aw) && put!(w,t+1), ph.work, ph.active_workers)
        # Prepare memory for next iteration
        push!(ph.subobjectives, zeros(nscenarios(ph)))
        push!(ph.finished, 0)
        # Log progress
        log!(ph)
    end
    # Just return a valid status for this iteration
    return :Valid
end
