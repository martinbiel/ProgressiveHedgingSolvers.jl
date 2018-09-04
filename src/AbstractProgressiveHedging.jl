abstract type AbstractProgressiveHedgingSolver{T <: Real, A <: AbstractVector, S <: LQSolver} <: AbstractStructuredModel end

nscenarios(ph::AbstractProgressiveHedgingSolver) = ph.nscenarios

# Initialization #
# ======================================================================== #
function init!(ph::AbstractProgressiveHedgingSolver,subsolver::AbstractMathProgSolver)
    # Initialize progress meter
    ph.progress.thresh = ph.parameters.τ
    # Finish initialization based on solver traits
    init_subproblems!(ph,subsolver)
end
# ======================================================================== #

# Functions #
# ======================================================================== #
function set_params!(ph::AbstractProgressiveHedgingSolver; kwargs...)
    for (k,v) in kwargs
        setfield!(ph.parameters,k,v)
    end
end

function current_objective_value(ph::AbstractProgressiveHedgingSolver,Qs::AbstractVector)
    return sum(Qs)
end
current_objective_value(ph) = current_objective_value(ph,ph.subobjectives)

function get_decision(ph::AbstractProgressiveHedgingSolver)
    return ph.ξ
end

function get_objective_value(ph::AbstractProgressiveHedgingSolver)
    if !isempty(ph.Q_history)
        return ph.Q_history[end]
    else
        return calculate_objective_value(ph)
    end
end

function iterate!(ph::AbstractProgressiveHedgingSolver)
    # Resolve all subproblems at the current optimal solution
    Q = resolve_subproblems!(ph)
    if Q == Inf
        return :Infeasible
    elseif Q == -Inf
        return :Unbounded
    end
    ph.solverdata.Q = Q
    # Update iterate
    update_iterate!(ph)
    # Log progress
    log!(ph)
    # Check optimality
    if check_optimality(ph)
        # Optimal
        return :Optimal
    end
    # Just return a valid status for this iteration
    return :Valid
end

function log!(ph::AbstractProgressiveHedgingSolver)
    @unpack Q,δ = ph.solverdata
    push!(ph.Q_history,Q)
    ph.solverdata.iterations += 1

    if ph.parameters.log
        ProgressMeter.update!(ph.progress,δ,
                              showvalues = [
                                  ("Objective",Q),
                                  ("δ",δ)
                              ])
    end
end

function check_optimality(ph::AbstractProgressiveHedgingSolver)
    @unpack τ = ph.parameters
    @unpack δ = ph.solverdata
    return δ <= τ
end
# ======================================================================== #
function show(io::IO, ph::AbstractProgressiveHedgingSolver)
    println(io,typeof(ph).name.name)
    println(io,"State:")
    show(io,ph.solverdata)
    println(io,"Parameters:")
    show(io,ph.parameters)
end

function show(io::IO, ::MIME"text/plain", ph::AbstractProgressiveHedgingSolver)
    show(io,ph)
end
# ======================================================================== #

# Plot recipe #
# ======================================================================== #
@recipe f(ph::AbstractProgressiveHedgingSolver) = ph,-1
@recipe function f(ph::AbstractProgressiveHedgingSolver, time::Real)
    length(ph.Q_history) > 0 || error("No solution data. Has solver been run?")
    Qmin = minimum(ph.Q_history)
    Qmax = maximum(ph.Q_history)
    increment = std(ph.Q_history)

    linewidth --> 4
    linecolor --> :black
    tickfontsize := 14
    tickfontfamily := "sans-serif"
    guidefontsize := 16
    guidefontfamily := "sans-serif"
    titlefontsize := 22
    titlefontfamily := "sans-serif"
    xlabel := time == -1 ? "Iteration" : "Time [s]"
    ylabel := "Q"
    ylims --> (Qmin-increment,Qmax+increment)
    if time == -1
        xlims --> (1,length(ph.Q_history)+1)
        xticks --> 1:5:length(ph.Q_history)
    else
        xlims --> (0,time)
        xticks --> linspace(0,time,ceil(Int,length(ph.Q_history)/5))
    end
    yticks --> Qmin:increment:Qmax
    xformatter := (d) -> @sprintf("%.1f",d)
    yformatter := (d) -> begin
        if abs(d) <= sqrt(eps())
            "0.0"
        elseif (log10(abs(d)) < -2.0 || log10(abs(d)) > 3.0)
            @sprintf("%.4e",d)
        elseif log10(abs(d)) > 2.0
            @sprintf("%.1f",d)
        else
            @sprintf("%.2f",d)
        end
    end

    @series begin
        label --> "Q"
        seriescolor --> :black
        if time == -1
            1:1:length(ph.Q_history),ph.Q_history
        else
            linspace(0,time,length(ph.Q_history)),ph.Q_history
        end
    end
end
# ======================================================================== #
