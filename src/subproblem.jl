struct SubProblem{T <: Real, A <: AbstractVector, S <: LQSolver}
    id::Int
    π::T
    r::T
    solver::S
    c::A
    x::A
    y::A
    ρ::A
    optimvector::A

    function (::Type{SubProblem})(model::JuMP.Model,id::Integer,π::AbstractFloat,r::AbstractFloat,xdim::Integer,optimsolver::AbstractMathProgSolver)
        solver = LQSolver(model,optimsolver)
        solver()
        optimvector = getsolution(solver)
        x₀ = optimvector[1:xdim]
        y₀ = optimvector[xdim+1:end]
        T = promote_type(eltype(optimvector),Float32)
        c_ = convert(AbstractVector{T},JuMP.prepAffObjective(model))
        c_ *= model.objSense == :Min ? 1 : -1
        x₀_ = convert(AbstractVector{T},x₀)
        y₀_ = convert(AbstractVector{T},y₀)
        optimvector_ = convert(AbstractVector{T},optimvector)
        A = typeof(x₀_)
        subproblem = new{T,A,typeof(solver)}(id,
                                             π,
                                             r,
                                             solver,
                                             c_,
                                             x₀_,
                                             y₀_,
                                             zeros(x₀_),
                                             optimvector_)
        return subproblem
    end

    function (::Type{SubProblem})(model::JuMP.Model,id::Integer,π::AbstractFloat,r::AbstractFloat,x₀::AbstractVector,y₀::AbstractVector,optimsolver::AbstractMathProgSolver)
        T = promote_type(eltype(x₀),eltype(y₀),Float32)
        c_ = convert(AbstractVector{T},JuMP.prepAffObjective(model))
        c_ *= model.objSense == :Min ? 1 : -1
        x₀_ = convert(AbstractVector{T},x₀)
        y₀_ = convert(AbstractVector{T},y₀)
        A = typeof(x₀_)
        solver = LQSolver(model,optimsolver)
        subproblem = new{T,A,typeof(solver)}(id,
                                             π,
                                             r,
                                             solver,
                                             c_,
                                             x₀_,
                                             y₀_,
                                             zeros(x₀_),
                                             [x₀_...,y₀_...])
        return subproblem
    end
end

function update_primal!(subproblem::SubProblem,ξ::AbstractVector)
    add_penalty!(subproblem,ξ)
end
update_primals!(subproblems::Vector{<:SubProblem},ξ::AbstractVector) = map(prob -> update_primal!(prob,ξ),subproblems)

function update_dual!(subproblem::SubProblem,ξ::AbstractVector)
    subproblem.ρ[:] = subproblem.ρ + subproblem.r*(subproblem.x - ξ)
end
update_duals!(subproblems::Vector{<:SubProblem},ξ::AbstractVector) = map(prob -> update_dual!(prob,ξ),subproblems)

function get_solution(subproblem::SubProblem)
    return copy(subproblem.y), getredcosts(subproblem.solver)[length(subproblem.x)+1:end],getduals(subproblem.solver)
end

function add_penalty!(subproblem::SubProblem,ξ::AbstractVector)
    model = subproblem.solver.lqmodel
    # Linear part
    c = copy(subproblem.c)
    c[1:length(ξ)] += subproblem.ρ
    c[1:length(ξ)] -= subproblem.r*ξ
    setobj!(model,c)
    # Quadratic part
    qidx = collect(1:length(subproblem.optimvector))
    qval = zeros(length(subproblem.optimvector))
    qval[1:length(ξ)] = 1.0
    if applicable(setquadobj!,model,qidx,qidx,qval)
        setquadobj!(model,qidx,qidx,qval)
    else
        error("Setting a quadratic penalty requires a solver that handles quadratic objectives")
    end
    # if subproblm.linearize
    #     ncols = lshaped.structuredmodel.numCols
    #     tidx = ncols+nscenarios(lshaped)+1
    #     j = lshaped.solverdata.regularizerindex
    #     if j == -1
    #         for i in 1:ncols
    #             addconstr!(model,[i,tidx],[-α,1],-α*ξ[i],Inf)
    #             addconstr!(model,[i,tidx],[-α,-1],-Inf,-ξ[i])
    #         end
    #     else
    #         for i in j:j+2*ncols-1
    #             delconstrs!(model,i)
    #         end
    #         for i in 1:ncols
    #             addconstr!(model,[i,tidx],[-α,1],-ξ[i],Inf)
    #             addconstr!(model,[i,tidx],[-α,-1],-Inf,-ξ[i])
    #         end
    #     end
    #     lshaped.solverdata.regularizerindex = length(lshaped.structuredmodel.linconstr)+length(lshaped.cuts)+1
    #     if hastrait(lshaped,HasLevels)
    #         lshaped.solverdata.regularizerindex += 1
    #     end
    # else
    #     # Linear part
    #     c[1:length(ξ)] -= α*ξ
    #     setobj!(model,c)
    #     # Quadratic part
    #     qidx = collect(1:length(ξ)+lshaped.nscenarios)
    #     qval = fill(α,length(lshaped.ξ))
    #     append!(qval,zeros(lshaped.nscenarios))
    #     if applicable(setquadobj!,model,qidx,qidx,qval)
    #         setquadobj!(model,qidx,qidx,qval)
    #     else
    #         error("Setting a quadratic penalty requires a solver that handles quadratic objectives")
    #     end
    # end
end

function (subproblem::SubProblem)()
    subproblem.solver(subproblem.optimvector)
    solvestatus = status(subproblem.solver)
    if solvestatus == :Optimal
        xdim = length(subproblem.x)
        subproblem.optimvector[:] = getsolution(subproblem.solver)
        subproblem.x[:] = subproblem.optimvector[1:xdim]
        subproblem.y[:] = subproblem.optimvector[xdim+1:end]
        return subproblem.π*subproblem.c⋅subproblem.optimvector
    elseif solvestatus == :Infeasible
        return Inf
    elseif solvestatus == :Unbounded
        return -Inf
    else
        error(@sprintf("Subproblem %d was not solved properly, returned status code: %s",subproblem.id,string(solvestatus)))
    end
end
