@scenario Integer = begin
    ξ₁::Int
    ξ₂::Int

    @expectation begin
        return IntegerScenario(sum([round(Int, probability(s)*s.ξ₁) for s in scenarios]),
                               sum([round(Int, probability(s)*s.ξ₂) for s in scenarios]))
    end
end
s₁ = IntegerScenario(2, 2, probability = 0.5)
s₂ = IntegerScenario(4, 3, probability = 0.5)

integer = StochasticProgram([s₁,s₂])

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
    @constraint(model, y₁ + 2*y₂ <= s.ξ₁-x₁)
    @constraint(model, y₁ <= s.ξ₂-x₂)
end

push!(problems, (integer, "Integer"))
