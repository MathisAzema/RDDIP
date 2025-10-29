function bin_extensive_neutral(instance; silent=true,  force::Float64=1.0, S::Int64=5, batch=1, gap=gap, timelimit=10)
    """
    Solve the two stage SUC with 3-bin extensive formulation
    """

    model = Model(instance.optimizer)
    if silent
        set_silent(model)
    end

    T= instance.TimeHorizon
    thermal_units_name=keys(instance.Thermalunits)
    thermal_units=values(instance.Thermalunits)

    T= instance.TimeHorizon
    thermal_units_name=keys(instance.Thermalunits)
    thermal_units=values(instance.Thermalunits)

    Next=instance.Next
    Buses=1:size(Next)[1]
    @variable(model, power[unit in thermal_units_name, t in 0:T, s in 1:S]>=0)
    @variable(model, power_shedding[b in Buses, t in 0:T, s in 1:S]>=0)
    @variable(model, power_curtailement[b in Buses, t in 0:T, s in 1:S]>=0)
    @variable(model, is_on[unit in thermal_units_name, t in 0:T], Bin)
    @variable(model, start_up[unit in thermal_units_name, t in 1:T], Bin)
    @variable(model, start_down[unit in thermal_units_name, t in 1:T], Bin)

    @variable(model, thermal_fuel_cost[s in 1:S]>=0)
    @variable(model, thermal_fixed_cost>=0)
    @variable(model, thermal_cost>=0)

    @constraint(model,  thermal_cost>=sum(thermal_fuel_cost[s] for s in 1:S)/S)

    thermal_unit_commit_constraints(model, instance)
    thermal_unit_capacity_constraints_scenarios(model, instance, S)

    @variable(model, θ[b in Buses, t in 1:T, s in 1:S])
    @variable(model, flow[b in Buses, bp in Next[b], t in 1:T, s in 1:S])
    flow_constraints_scenarios(model,instance,S)

    BusWind=instance.BusWind
    NumWindfarms=length(BusWind)
    Wpower=instance.WGscenario
    List_scenario=instance.Training_set[batch]

    @constraint(model,  demand[t in 1:T, b in Buses, s in 1:S], sum(power[unit.name, t, s] for unit in thermal_units if unit.Bus==b)+power_shedding[b,t,s]-power_curtailement[b,t,s]==instance.Demandbus[b][t]+sum(flow[b,bp,t,s] for bp in Next[b])-force * sum(Wpower[w][t,List_scenario[s]] for w in 1:NumWindfarms if BusWind[w]==b))

    @objective(model, Min, thermal_fixed_cost+thermal_cost)
  
    set_optimizer_attribute(model, "TimeLimit", timelimit)
    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "MIPGap", gap)

    start = time()
    optimize!(model)
    computation_time = time() - start

    feasibleSolutionFound = primal_status(model) == MOI.FEASIBLE_POINT

    if feasibleSolutionFound

        solution_is_on=JuMP.value.(is_on)
        solution_start_up=JuMP.value.(start_up)
        solution_start_down=JuMP.value.(start_down)

        solution_thermal=[solution_is_on, solution_start_up, solution_start_down]

        return instance.name, computation_time, objective_value(model), objective_bound(model), solution_thermal, S, batch,  gap, force, value.(power)

    else
        return nothing
    end
end

function extensive_AVAR(instance; silent=true,  α=0, force::Float64=1.0, S::Int64=5, batch=1, gap=gap, timelimit=10)
    """
    Solve the two stage risk-averse SUC with 3-bin extensive formulation
    """

    model = Model(instance.optimizer)
    if silent
        set_silent(model)
    end

    T= instance.TimeHorizon
    thermal_units_name=keys(instance.Thermalunits)
    thermal_units=values(instance.Thermalunits)

    T= instance.TimeHorizon
    thermal_units_name=keys(instance.Thermalunits)
    thermal_units=values(instance.Thermalunits)

    Next=instance.Next
    Buses=1:size(Next)[1]
    @variable(model, power[unit in thermal_units_name, t in 0:T, s in 1:S]>=0)
    @variable(model, power_shedding[b in Buses, t in 0:T, s in 1:S]>=0)
    @variable(model, power_curtailement[b in Buses, t in 0:T, s in 1:S]>=0)
    @variable(model, is_on[unit in thermal_units_name, t in 0:T], Bin)
    @variable(model, start_up[unit in thermal_units_name, t in 1:T], Bin)
    @variable(model, start_down[unit in thermal_units_name, t in 1:T], Bin)

    @variable(model, thermal_fuel_cost[s in 1:S]>=0)
    @variable(model, thermal_fuel_cost_pos[s in 1:S]>=0)
    @variable(model, thermal_fixed_cost>=0)
    @variable(model, thermal_cost>=0)
    @variable(model, z_AVAR>=0)

    @constraint(model,  thermal_cost>=sum(thermal_fuel_cost_pos[s] for s in 1:S)/S)
    @constraint(model,  [s in 1:S], thermal_fuel_cost_pos[s]>=thermal_fuel_cost[s]-z_AVAR)

    thermal_unit_commit_constraints(model, instance)
    thermal_unit_capacity_constraints_scenarios(model, instance, S)

    @variable(model, θ[b in Buses, t in 1:T, s in 1:S])
    @variable(model, flow[b in Buses, bp in Next[b], t in 1:T, s in 1:S])
    flow_constraints_scenarios(model,instance,S)

    BusWind=instance.BusWind
    NumWindfarms=length(BusWind)
    Wpower=instance.WGscenario
    List_scenario=instance.Training_set[batch]

    @constraint(model,  [t in 1:T, b in Buses, s in 1:S], sum(power[unit.name, t, s] for unit in thermal_units if unit.Bus==b)+power_shedding[b,t,s]-power_curtailement[b,t,s]==instance.Demandbus[b][t]+sum(flow[b,bp,t,s] for bp in Next[b])-force * sum(Wpower[w][t,List_scenario[s]] for w in 1:NumWindfarms if BusWind[w]==b))

    @objective(model, Min, thermal_fixed_cost+z_AVAR+thermal_cost/(1-α))
  
    set_optimizer_attribute(model, "TimeLimit", timelimit)
    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "MIPGap", gap)

    start = time()
    optimize!(model)
    computation_time = time() - start

    feasibleSolutionFound = primal_status(model) == MOI.FEASIBLE_POINT

    if feasibleSolutionFound

        solution_is_on=JuMP.value.(is_on)
        solution_start_up=JuMP.value.(start_up)
        solution_start_down=JuMP.value.(start_down)

        solution_thermal=[solution_is_on, solution_start_up, solution_start_down]

        return instance.name, computation_time, objective_value(model), objective_bound(model), solution_thermal, S, batch,  gap, force

    else
        return nothing
    end
end

function extended_extensive_neutral(instance; silent=true,  force::Float64=1.0, S::Int64=5, batch=1, gap=gap, timelimit=10)
    """
    Solve the two stage SUC with extended extensive formulation
    """

    model = Model(instance.optimizer)
    if silent
        set_silent(model)
    end

    T= instance.TimeHorizon
    thermal_units_name=keys(instance.Thermalunits)
    thermal_units=values(instance.Thermalunits)

    T= instance.TimeHorizon
    thermal_units_name=keys(instance.Thermalunits)
    thermal_units=values(instance.Thermalunits)

    Next=instance.Next
    Buses=1:size(Next)[1]

    T= instance.TimeHorizon
    N=instance.N
    thermal_units=instance.Thermalunits

    @variable(model, power[i in 1:N, t in 0:T, s in 1:S]>=0)
    @variable(model, power_shedding[b in Buses, t in 0:T, s in 1:S]>=0)
    @variable(model, power_curtailement[b in Buses, t in 0:T, s in 1:S]>=0)
    @variable(model, is_on[i in 1:N, t in 0:T], Bin)
    @variable(model, start_up[i in 1:N, t in 1:T], Bin)
    @variable(model, start_down[i in 1:N, t in 1:T], Bin)
    @variable(model, gamma[i in 1:N, (a,b) in instance.Thermalunits[i].intervals]>=0)
    @constraint(model, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if a<=t<b)==is_on[unit.name,t])
    @constraint(model, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if a==t)==start_up[unit.name,t])
    @constraint(model, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if b==t)==start_down[unit.name,t])

    @variable(model, thermal_fuel_cost[s in 1:S]>=0)
    @variable(model, thermal_fixed_cost>=0)
    @variable(model, thermal_cost>=0)

    @constraint(model,  thermal_cost>=sum(thermal_fuel_cost[s] for s in 1:S)/S)

    thermal_unit_commit_constraints_extended(model, instance)
    thermal_unit_capacity_constraints_scenarios(model, instance, S)

    @variable(model, θ[b in Buses, t in 1:T, s in 1:S])
    @variable(model, flow[b in Buses, bp in Next[b], t in 1:T, s in 1:S])
    flow_constraints_scenarios(model,instance,S)

    BusWind=instance.BusWind
    NumWindfarms=length(BusWind)
    Wpower=instance.WGscenario
    List_scenario=instance.Training_set[batch]

    @constraint(model,  [t in 1:T, b in Buses, s in 1:S], sum(power[unit.name, t, s] for unit in thermal_units if unit.Bus==b)+power_shedding[b,t,s]-power_curtailement[b,t,s]==instance.Demandbus[b][t]+sum(flow[b,bp,t,s] for bp in Next[b])-force * sum(Wpower[w][t,List_scenario[s]] for w in 1:NumWindfarms if BusWind[w]==b))

    @objective(model, Min, thermal_fixed_cost+thermal_cost)
  
    set_optimizer_attribute(model, "TimeLimit", timelimit)
    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "MIPGap", gap)

    start = time()
    optimize!(model)
    computation_time = time() - start

    feasibleSolutionFound = primal_status(model) == MOI.FEASIBLE_POINT

    if feasibleSolutionFound

        solution_is_on=JuMP.value.(is_on)
        solution_start_up=JuMP.value.(start_up)
        solution_start_down=JuMP.value.(start_down)

        solution_thermal=[solution_is_on, solution_start_up, solution_start_down]

        return instance.name, computation_time, objective_value(model), objective_bound(model), solution_thermal, S, batch,  gap, force

    else
        return nothing
    end
end