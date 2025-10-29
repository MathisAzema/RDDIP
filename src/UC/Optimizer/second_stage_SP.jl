struct oracleResults
    objective_value::Float64
    intercept3bin::Float64
    interceptE::Float64
    muup::Matrix{Float64}
    mudown::Matrix{Float64}
    mumax::Matrix{Float64}
    mumin::Matrix{Float64}
    ν::Matrix{Float64}
    status::Bool
    computation_time::Float64
end

function second_stage_SP_extended(instance, oracle_pb; batch=1, scenario=1, force=1, price=nothing)

    """
    Solve the dual of the second-stage problem (we don't need to get the mumax and mumin values)
    """

    T= instance.TimeHorizon
    thermal_units=instance.Thermalunits
    Next=instance.Next
    Buses=1:size(Next)[1] 

    μꜛ=oracle_pb[:μꜛ]
    μꜜ=oracle_pb[:μꜜ]
    ν=oracle_pb[:ν]
    λ=oracle_pb[:λ]

    BusWind=instance.BusWind
    NumWindfarms=length(BusWind)
    Wpower=instance.WGscenario
    List_scenario=instance.Training_set[batch]

    if price!=nothing
        @constraint(oracle_pb, [b in Buses, t in 1:T], ν[b,t]==price[b,t])
    end

    Netdemand = [instance.Demandbus[b][t] - force * sum(Wpower[w][t, List_scenario[scenario]] for w in 1:NumWindfarms if BusWind[w] == b;init=0) for b in Buses, t in 1:T]
    set_objective_coefficient.(oracle_pb,ν, Netdemand)

    start = time()
    optimize!(oracle_pb)
    computation_time = time() - start

    status = JuMP.termination_status(oracle_pb)!=MOI.DUAL_INFEASIBLE

    μꜛₖ=JuMP.value.(μꜛ)
    μꜜₖ=JuMP.value.(μꜜ)
    λₖ=convert(Vector{Float64}, JuMP.value.(λ))
    νₖ=convert(Matrix{Float64}, JuMP.value.(ν))


    βₖ2=sum(νₖ[b,t]*Netdemand[b,t] for b in Buses for t in 1:T)+value.(oracle_pb[:network_cost])

    βₖ=sum(λₖ[unit.name]*unit.InitialPower for unit in thermal_units if unit.InitialPower!=nothing; init=0)+βₖ2

    mumax=zeros(Float64, size(μꜛₖ))
    mumin=zeros(Float64, size(μꜛₖ))

    return oracleResults(objective_value(oracle_pb), βₖ, βₖ2, μꜛₖ, μꜜₖ,mumax, mumin, νₖ, status, computation_time)
end

function second_stage_SP_3bin(instance, oracle_pb; batch=1, scenario=1, force=1, price=nothing)
    """
    Solve the dual of the second-stage problem
    """

    T= instance.TimeHorizon
    thermal_units=instance.Thermalunits
    Next=instance.Next
    Buses=1:size(Next)[1] 

    μꜛ=oracle_pb[:μꜛ]
    μꜜ=oracle_pb[:μꜜ]
    ν=oracle_pb[:ν]
    λ=oracle_pb[:λ]

    BusWind=instance.BusWind
    NumWindfarms=length(BusWind)
    Wpower=instance.WGscenario
    List_scenario=instance.Training_set[batch]

    if price!=nothing
        @constraint(oracle_pb, [b in Buses, t in 1:T], ν[b,t]==price[b,t])
    end

    Netdemand = [instance.Demandbus[b][t] - force * sum(Wpower[w][t, List_scenario[scenario]] for w in 1:NumWindfarms if BusWind[w] == b;init=0) for b in Buses, t in 1:T]
    set_objective_coefficient.(oracle_pb,ν, Netdemand)

    start = time()
    optimize!(oracle_pb)
    computation_time = time() - start

    status = termination_status(oracle_pb)!=MOI.DUAL_INFEASIBLE

    μꜛₖ=JuMP.value.(μꜛ)
    μꜜₖ=JuMP.value.(μꜜ)
    mumin=JuMP.value.(oracle_pb[:μₘᵢₙ])
    mumax=JuMP.value.(oracle_pb[:μₘₐₓ])

    λₖ=convert(Vector{Float64}, JuMP.value.(λ))
    νₖ=convert(Matrix{Float64}, JuMP.value.(ν))


    βₖ2=sum(νₖ[b,t]*Netdemand[b,t] for b in Buses for t in 1:T)+value.(oracle_pb[:network_cost])

    βₖ=sum(λₖ[unit.name]*unit.InitialPower for unit in thermal_units if unit.InitialPower!=nothing; init=0)+βₖ2

    return oracleResults(objective_value(oracle_pb), βₖ, βₖ2, μꜛₖ, μꜜₖ,mumax, mumin, νₖ, status, computation_time)
end