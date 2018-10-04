@everywhere begin
    struct IntegerScenario <: AbstractScenarioData
        π::Probability
        ξ::Vector{Int}
    end

    function StochasticPrograms.expected(sds::Vector{IntegerScenario})
        sd = IntegerScenario(1,sum([s.π*s.ξ for s in sds]))
    end
end

s1 = IntegerScenario(0.5,[2,2])
s2 = IntegerScenario(0.5,[4,3])

sds = [s1,s2]

integer = StochasticProgram(sds)

@first_stage integer = begin
    @variable(model, x₁, Bin)
    @variable(model, x₂, Bin)
end

@second_stage integer = begin
    @decision x₁ x₂
    s = scenario
    @variable(model, y₁ >= 0, Int)
    @variable(model, y₂ >= 0, Int)
    @objective(model, Min, -2*y₁ - 3*y₂)
    @constraint(model, y₁ + 2*y₂ <= s.ξ[1]-x₁)
    @constraint(model, y₁ <= s.ξ[2]-x₂)
end

push!(problems,(integer,"Integer"))
