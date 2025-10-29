function flow_constraints(model, instance)
    """
    Define Flow constraints in the network
    """
    θ=model[:θ]
    flow=model[:flow]

    T= instance.TimeHorizon
    Lines=values(instance.Lines)
    Next=instance.Next
    Buses=1:size(Next)[1]

    @constraint(model, [line in Lines, t in 1:T], flow[line.b1,line.b2,t]<=line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.b1,line.b2,t]>=-line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.b1,line.b2,t]==line.B12*(θ[line.b1,t]-θ[line.b2,t]))
end

function flow_constraints_scenarios(model, instance, S)
    """
    Define Flow constraints in the network with S scenarios
    """
    θ=model[:θ]
    flow=model[:flow]

    T= instance.TimeHorizon
    Lines=values(instance.Lines)

    @constraint(model, [line in Lines, t in 1:T, s in 1:S], flow[line.b1,line.b2,t,s]<=line.Fmax)
    @constraint(model, [line in Lines, t in 1:T, s in 1:S], flow[line.b1,line.b2,t,s]>=-line.Fmax)
    @constraint(model, [line in Lines, t in 1:T, s in 1:S], flow[line.b1,line.b2,t,s]==line.B12*(θ[line.b1,t,s]-θ[line.b2,t,s]))
end