function set_intervals(MinUpTime, InitUpDownTime, InitialPower, MaxPower, MinPower, DeltaRampDown)
    """
    Define all the possible intervals [a,b] respecting the initial conditions and the the MinUp constraints
    """
    bmax=25
    # bmax=24+ceil((MaxPower-MinPower)/DeltaRampDown)+1
    intervals=[[a,b] for a in 1:24 for b in a+MinUpTime:bmax]
    if InitUpDownTime!=nothing
        if InitUpDownTime>=0
            for b in max(1,-InitUpDownTime+1+MinUpTime, Int64(1+ceil((InitialPower-MinPower)/DeltaRampDown))):bmax
                push!(intervals, [-InitUpDownTime+1,b])
            end
        end
    end
    return intervals
end

function thermal_unit_commit_constraints(model, instance)
    """
    Define first-stage constraints in the 3-bin formulation
    """
    is_on=model[:is_on]
    start_up=model[:start_up]
    start_down=model[:start_down]
    thermal_fixed_cost=model[:thermal_fixed_cost]

    T= instance.TimeHorizon
    thermal_units=values(instance.Thermalunits)

    @constraint(model,  [unit in thermal_units, t in 1:T], is_on[unit.name, t]-is_on[unit.name, t-1]==start_up[unit.name, t]-start_down[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T], start_up[unit.name, t]<=is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T], start_down[unit.name, t]<=1-is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime!=nothing], sum(start_up[unit.name, τ] for τ in max(1, t-unit.MinUpTime+1):t)+1*(t<unit.MinUpTime-unit.InitUpDownTime+1)*(unit.InitUpDownTime>0) <= is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime!=nothing], sum(start_down[unit.name, τ] for τ in max(1, t-unit.MinDownTime+1):t)+1*(t<unit.MinDownTime+unit.InitUpDownTime+1)*(unit.InitUpDownTime<0) <= 1-is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime==nothing], sum(start_up[unit.name, τ] for τ in max(1, t-unit.MinUpTime+1):t) <= is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime==nothing], sum(start_down[unit.name, τ] for τ in max(1, t-unit.MinDownTime+1):t) <= 1-is_on[unit.name, t])

    @constraint(model,  [unit in thermal_units; unit.InitUpDownTime!=nothing], is_on[unit.name, 0]==(unit.InitUpDownTime>=0))

    @constraint(model,  [unit in thermal_units; unit.InitUpDownTime==nothing], is_on[unit.name, 0]==0)

    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_units for t in 1:T))

    for unit in thermal_units
        if (unit.InitialPower-unit.MinPower)/unit.DeltaRampDown <0
            limit=-1
        else
            limit=Int64(ceil((unit.InitialPower-unit.MinPower)/unit.DeltaRampDown))
        end
        for t in 0:limit
            @constraint(model, is_on[unit.name, t]==1)
        end
    end
end

function thermal_unit_commit_constraints_extended(model, instance)
    """
    Define first-stage constraints in the extended formulation
    """
    is_on=model[:is_on]
    start_up=model[:start_up]
    start_down=model[:start_down]
    thermal_fixed_cost=model[:thermal_fixed_cost]

    T= instance.TimeHorizon
    thermal_units=values(instance.Thermalunits)

    @constraint(model,  [unit in thermal_units, t in 1:T], is_on[unit.name, t]-is_on[unit.name, t-1]==start_up[unit.name, t]-start_down[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T], start_up[unit.name, t]<=is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T], start_down[unit.name, t]<=1-is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime!=nothing], sum(start_down[unit.name, τ] for τ in max(1, t-unit.MinDownTime+1):t)+1*(t<unit.MinDownTime+unit.InitUpDownTime+1)*(unit.InitUpDownTime<0) <= 1-is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 1:T; unit.InitUpDownTime==nothing], sum(start_down[unit.name, τ] for τ in max(1, t-unit.MinDownTime+1):t) <= 1-is_on[unit.name, t])

    @constraint(model,  [unit in thermal_units; unit.InitUpDownTime!=nothing], is_on[unit.name, 0]==(unit.InitUpDownTime>=0))

    @constraint(model,  [unit in thermal_units; unit.InitUpDownTime==nothing], is_on[unit.name, 0]==0)

    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_units for t in 1:T))

    for unit in thermal_units
        if (unit.InitialPower-unit.MinPower)/unit.DeltaRampDown <0
            limit=-1
        else
            limit=Int64(ceil((unit.InitialPower-unit.MinPower)/unit.DeltaRampDown))
        end
        for t in 0:limit
            @constraint(model, is_on[unit.name, t]==1)
        end
    end
end

function thermal_unit_capacity_constraints_scenarios(model, instance, S)
    """
    Define second-stage constraints in the extensive formulation
    """
    power=model[:power]
    is_on=model[:is_on]
    start_up=model[:start_up]
    start_down=model[:start_down]
    thermal_fuel_cost=model[:thermal_fuel_cost]
    power_shedding=model[:power_shedding]
    power_curtailement=model[:power_curtailement]

    T= instance.TimeHorizon
    Next=instance.Next
    Buses=1:size(Next)[1]
    thermal_units=values(instance.Thermalunits)

    @constraint(model,  [unit in thermal_units, t in 0:T, s in 1:S], power[unit.name, t, s]>=unit.MinPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 0:T, s in 1:S], power[unit.name, t, s]<=unit.MaxPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, s in 1:S; unit.InitialPower!=nothing], power[unit.name, 0, s]==unit.InitialPower)

    @constraint(model,  [unit in thermal_units, t in 1:T, s in 1:S], power[unit.name, t, s]-power[unit.name, t-1, s]<=(-unit.DeltaRampUp)*start_up[unit.name, t]+(unit.MinPower+unit.DeltaRampUp)*is_on[unit.name, t]-(unit.MinPower)*is_on[unit.name, t-1])
    @constraint(model,  [unit in thermal_units, t in 1:T, s in 1:S], power[unit.name, t-1, s]-power[unit.name, t, s]<=(-unit.DeltaRampDown)*start_down[unit.name, t]+(unit.MinPower+unit.DeltaRampDown)*is_on[unit.name, t-1]-(unit.MinPower)*is_on[unit.name, t])

    @constraint(model, [s in 1:S], thermal_fuel_cost[s]>=sum(unit.LinearTerm*power[unit.name, t, s] for unit in thermal_units for t in 1:T)+sum(SHEDDING_COST*power_shedding[b,t,s]+CURTAILEMENT_COST*power_curtailement[b,t,s] for b in Buses for t in 1:T))
end