
function CCG_algo(instance; silent=true, Γ=0, MaxIter = 2, gap = 0.5)
    T= instance.TimeHorizon
    N=instance.N    
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    Numlines=length(instance.Lines)
    thermal_units=instance.Thermalunits

    master_pb = master_RO_problem(instance; silent=silent)

    oracle_pb = oracle_RO_problem(instance; silent=silent, Γ=Γ)
    dual_var_is_on = oracle_pb.dual_var_is_on
    dual_var_start_up = oracle_pb.dual_var_start_up
    dual_var_start_down = oracle_pb.dual_var_start_down
    dual_var_uncertainty = oracle_pb.dual_var_uncertainty
    dual_demand = oracle_pb.dual_demand

    println(dual_demand)

    LB= 1
    UB= 1e9
    k = 0
    while 100*(UB - LB)/UB > gap && k < MaxIter

        k += 1
        JuMP.optimize!(master_pb)

        LB = JuMP.objective_bound(master_pb)

        println((JuMP.value(master_pb[:thermal_fixed_cost]), JuMP.value(master_pb[:thermal_cost])))

        solution_is_on = JuMP.value.(master_pb[:is_on])
        solution_start_up = JuMP.value.(master_pb[:start_up])
        solution_start_down = JuMP.value.(master_pb[:start_down])

        primal_obj = JuMP.objective_function(oracle_pb.model)

        JuMP.set_objective_function(
            oracle_pb.model,
            @expression(oracle_pb.model, primal_obj + sum(
                dual_var_is_on[i,t] * solution_is_on[i,t] for
                i in 1:N for t in 0:T)) + sum(
                dual_var_start_up[i,t] * solution_start_up[i,t] for
                i in 1:N for t in 1:T) + sum(
                dual_var_start_down[i,t] * solution_start_down[i,t] for
                i in 1:N for t in 1:T)
        )

        JuMP.optimize!(oracle_pb.model)
        UB = min(UB, JuMP.objective_value(master_pb) - JuMP.value(master_pb[:thermal_cost]) + JuMP.objective_bound(oracle_pb.model))

        solution_uncertainty = JuMP.value.(oracle_pb.model[:uncertainty])

        println([t for t in 1:T if solution_uncertainty[t] >= 0.5])

        println((k, LB, UB, 100*(UB - LB)/UB))

        println([JuMP.value(dual_var_uncertainty[t]) for t in 1:T])
        println([JuMP.value(dual_var_uncertainty[t])/(sum(instance.Demandbus[b][t]*1.96*0.025 for b in Buses)) for t in 1:T])
        println([JuMP.value.(dual_demand[t, b]) for t in 1:T for b in Buses])

        if 100*(UB - LB)/UB > gap && k < MaxIter

            JuMP.set_objective_function(oracle_pb.model, primal_obj)

            power_k = @variable(
                master_pb, [i in 1:N, t in 0:T],
                base_name = "power_$k",
            )
            power_shedding_k = @variable(
                master_pb, [b in Buses, t in 1:T],
                base_name = "power_shedding_$k",
            )
            power_curtailement_k = @variable(
                master_pb, [b in Buses, t in 1:T],
                base_name = "power_curtailement_$k",
            )

            flow_k = @variable(
                master_pb, [l in 1:Numlines, t in 1:T],
                base_name = "flow_$k",
            )

            θ_k = @variable(
                master_pb, [b in Buses, t in 1:T],
                base_name = "theta_$k",
            )

            @constraint(master_pb, [line in Lines, t in 1:T], flow_k[line.id,t]<=line.Fmax)
            @constraint(master_pb, [line in Lines, t in 1:T], flow_k[line.id,t]>=-line.Fmax)
            @constraint(master_pb, [line in Lines, t in 1:T], flow_k[line.id,t]==line.B12*(θ_k[line.b1,t]-θ_k[line.b2,t]))

            @constraint(master_pb,  [b in Buses, t in 1:T], power_shedding_k[b, t]>=0)
            @constraint(master_pb,  [b in Buses, t in 1:T], power_curtailement_k[b, t]>=0)

            @constraint(master_pb,  [unit in thermal_units, t in 0:T], power_k[unit.name, t]>=unit.MinPower*master_pb[:is_on][unit.name, t])
            @constraint(master_pb,  [unit in thermal_units, t in 0:T], power_k[unit.name, t]<=unit.MaxPower*master_pb[:is_on][unit.name, t])
            @constraint(master_pb,  [unit in thermal_units; unit.InitialPower!=nothing], power_k[unit.name, 0]==unit.InitialPower)

            @constraint(master_pb,  [unit in thermal_units, t in 1:T], power_k[unit.name, t]-power_k[unit.name, t-1]<=(-unit.DeltaRampUp)*master_pb[:start_up][unit.name, t]+(unit.MinPower+unit.DeltaRampUp)*master_pb[:is_on][unit.name, t]-(unit.MinPower)*master_pb[:is_on][unit.name, t-1])
            @constraint(master_pb,  [unit in thermal_units, t in 1:T], power_k[unit.name, t-1]-power_k[unit.name, t]<=(-unit.DeltaRampDown)*master_pb[:start_down][unit.name, t]+(unit.MinPower+unit.DeltaRampDown)*master_pb[:is_on][unit.name, t-1]-(unit.MinPower)*master_pb[:is_on][unit.name, t])

            @constraint(master_pb, master_pb[:thermal_cost] >= sum(unit.LinearTerm*power_k[unit.name, t] for unit in thermal_units for t in 1:T)+sum(SHEDDING_COST*power_shedding_k[b,t]+CURTAILEMENT_COST*power_curtailement_k[b,t] for b in Buses for t in 1:T))

            @constraint(master_pb,  [t in 1:T, b in Buses], sum(power_k[unit.name, t] for unit in thermal_units if unit.Bus==b)+power_shedding_k[b,t]-power_curtailement_k[b,t] + sum(flow_k[line.id, t] for line in Lines if line.b2==b) - sum(flow_k[line.id, t] for line in Lines if line.b1==b) == instance.Demandbus[b][t]*(1+solution_uncertainty[t]*1.96*0.025))
        end
    end
    # solution_power = Dict()
    # for i in 1:N
    #     solution_power[i] = [JuMP.value(JuMP.variable_by_name(master_pb, "power_1[$i,$t]")) for t in 0:T]
    # end
    # solution_power_shedding = [JuMP.value(JuMP.variable_by_name(master_pb, "power_shedding_1[$b,$t]")) for b in Buses for t in 1:T]
    # solution_power_curtailement = [JuMP.value(JuMP.variable_by_name(master_pb, "power_curtailement_1[$b,$t]")) for b in Buses for t in 1:T]
    # return master_pb, JuMP.value.(master_pb[:is_on]), JuMP.value(master_pb[:thermal_cost]), JuMP.value(master_pb[:thermal_fixed_cost]), solution_power, solution_power_shedding, solution_power_curtailement
    # println(master_pb)
    solution_is_on = JuMP.value.(master_pb[:is_on])
    solution_start_up = JuMP.value.(master_pb[:start_up])
    solution_start_down = JuMP.value.(master_pb[:start_down])
    solution = (solution_is_on, solution_start_up, solution_start_down)
    solution_uncertainty = JuMP.value.(oracle_pb.model[:uncertainty])
    return k, LB, UB, 100*(UB - LB)/UB, solution, solution_uncertainty
end