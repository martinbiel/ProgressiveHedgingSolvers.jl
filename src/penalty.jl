# ------------------------------------------------------------
# Penalty: Penalty parameter strategy
# ------------------------------------------------------------
@define_trait Penalty = begin
    Fixed    # Algorithm uses a fixed user-defined penalty parameter
    Adaptive # Algorithm uses an adaptive penalty parameter
end
@define_traitfn Penalty penalty(ph::AbstractProgressiveHedgingSolver)
@define_traitfn Penalty init_penalty!(ph::AbstractProgressiveHedgingSolver)
@define_traitfn Penalty update_penalty!(ph::AbstractProgressiveHedgingSolver)
# Fixed
# ------------------------------------------------------------
@implement_traitfn function penalty(ph::AbstractProgressiveHedgingSolver, Fixed)
    return ph.parameters.r
end
@implement_traitfn function init_penalty!(ph::AbstractProgressiveHedgingSolver, Fixed)
    nothing
end
@implement_traitfn function update_penalty!(ph::AbstractProgressiveHedgingSolver, Fixed)
    nothing
end
# Adaptive
# ------------------------------------------------------------
@implement_traitfn function penalty(ph::AbstractProgressiveHedgingSolver, Adaptive)
    return ph.solverdata.r
end
@implement_traitfn function init_penalty!(ph::AbstractProgressiveHedgingSolver, Adaptive)
    update_dual_gap!(ph)
    @unpack δ₂ = ph.solverdata
    @unpack ζ = ph.parameters
    ph.solverdata.r = max(1., 2*ζ*abs(calculate_objective_value(ph)))/max(1., δ₂)
end
@implement_traitfn function update_penalty!(ph::AbstractProgressiveHedgingSolver, Adaptive)
    @unpack r, δ₁, δ₂ = ph.solverdata
    @unpack γ₁, γ₂, γ₃, σ, α, θ, ν, β, η = ph.parameters

    δ₂_prev = length(ph.dual_gaps) > 0 ? ph.dual_gaps[end] : Inf

    μ = if δ₁/norm(ph.ξ,2)^2 >= γ₁
        if (δ₁-δ₂)/(1e-10 + δ₂) > γ₂
            α
        elseif (δ₂-δ₁)/(1e-10 + δ₁) > γ₃
            θ
        else
            1.
        end
    elseif δ₂ > δ₂_prev
        if (δ₂-δ₂_prev)/δ₂_prev > ν
            β
        else
            1.
        end
    else
        η
    end
    ph.solverdata.r = μ*r
end
