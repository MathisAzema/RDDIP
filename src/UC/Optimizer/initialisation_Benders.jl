struct UCOptions
    master_problem::Function
    oracle_problem::Function
    second_stage::Function
    add_cut::Function
    _add_feasibility_cuts::Function
    _add_optimality_cuts::Function
end

function master_AVAR_problem(instance; silent=true, S::Int64, α=0)

    """
    Initial master problem in the AVAR extended formulation
    """

    model = Model(instance.optimizer)
    if silent
        set_silent(model)
    end

    set_optimizer_attribute(model, "Threads", 1)

    T= instance.TimeHorizon
    N=instance.N
    thermal_units=instance.Thermalunits

    @variable(model, is_on[i in 1:N, t in 0:T], Bin)
    @variable(model, start_up[i in 1:N, t in 1:T], Bin)
    @variable(model, start_down[i in 1:N, t in 1:T], Bin)

    @variable(model, gamma[i in 1:N, (a,b) in instance.Thermalunits[i].intervals]>=0)
    @constraint(model, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if a<=t<b)==is_on[unit.name,t])
    @constraint(model, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if a==t)==start_up[unit.name,t])
    @constraint(model, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if b==t)==start_down[unit.name,t])

    @variable(model, thermal_fuel_cost[s in 1:S]>=0)
    @variable(model, thermal_fuel_cost_pos[s in 1:S]>=0)
    @variable(model, thermal_fixed_cost>=0)
    @variable(model, thermal_cost>=0)
    @variable(model, z_AVAR>=0)

    thermal_unit_commit_constraints(model, instance)
    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_units for t in 1:T))

    @constraint(model,  thermal_cost>=sum(thermal_fuel_cost_pos[s] for s in 1:S)/S)
    @constraint(model,  [s in 1:S], thermal_fuel_cost_pos[s]>=thermal_fuel_cost[s]-z_AVAR)

    @objective(model, Min, thermal_fixed_cost+z_AVAR+thermal_cost/(1-α))
    
    return model
end

function master_SP_problem(instance; silent=true, S::Int64)
    """
    Initial master problem in the risk neutral extended formulation
    """

    model = Model(instance.optimizer)
    if silent
        set_silent(model)
    end

    set_optimizer_attribute(model, "Threads", 1)

    T= instance.TimeHorizon
    N=instance.N
    thermal_units=instance.Thermalunits

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

    thermal_unit_commit_constraints(model, instance)
    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_units for t in 1:T))

    @constraint(model,  thermal_cost>=sum(thermal_fuel_cost[s] for s in 1:S)/S)

    @objective(model, Min, thermal_fixed_cost+thermal_cost)
    
    return model
end

function master_3BD_problem(instance; silent=true, S::Int64)
    """
    Initial master problem in the risk neutral 3-bin formulation
    """

    model = Model(instance.optimizer)
    if silent
        set_silent(model)
    end

    set_optimizer_attribute(model, "Threads", 1)

    T= instance.TimeHorizon
    N=instance.N

    @variable(model, is_on[i in 1:N, t in 0:T], Bin)
    @variable(model, start_up[i in 1:N, t in 1:T], Bin)
    @variable(model, start_down[i in 1:N, t in 1:T], Bin)

    @variable(model, thermal_fuel_cost[s in 1:S]>=0)
    @variable(model, thermal_fixed_cost>=0)
    @variable(model, thermal_cost>=0)

    thermal_unit_commit_constraints(model, instance)
    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_units for t in 1:T))

    @constraint(model,  thermal_cost>=sum(thermal_fuel_cost[s] for s in 1:S)/S)

    @objective(model, Min, thermal_fixed_cost+thermal_cost)
    
    return model
end

function master_RO_problem(instance; silent=true)
    """
    Initial master problem in the risk neutral 3-bin formulation
    """

    model = Model(instance.optimizer)
    if silent
        set_silent(model)
    end

    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "TimeLimit", 5)

    T= instance.TimeHorizon
    N=instance.N    
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    Numlines=length(instance.Lines)
    thermal_units=instance.Thermalunits

    @variable(model, is_on[i in 1:N, t in 0:T], Bin)
    @variable(model, start_up[i in 1:N, t in 1:T], Bin)
    @variable(model, start_down[i in 1:N, t in 1:T], Bin)

    @variable(model, thermal_cost>=0)
    @variable(model, thermal_fixed_cost>=0)

    thermal_unit_commit_constraints(model, instance)
    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_units for t in 1:T))

    @variable(model, power[i in 1:N, t in 0:T]>=0)
    @variable(model, power_shedding[b in Buses, t in 1:T]>=0)
    @variable(model, power_curtailement[b in Buses, t in 1:T]>=0)

    @variable(model, flow[l in 1:Numlines, t in 1:T])
    @variable(model, θ[b in Buses, t in 1:T])

    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]<=line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]>=-line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]==line.B12*(θ[line.b1,t]-θ[line.b2,t]))

    @constraint(model,  [unit in thermal_units, t in 0:T], power[unit.name, t]>=unit.MinPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 0:T], power[unit.name, t]<=unit.MaxPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units; unit.InitialPower!=nothing], power[unit.name, 0]==unit.InitialPower)

    @constraint(model,  [unit in thermal_units, t in 1:T], power[unit.name, t]-power[unit.name, t-1]<=(-unit.DeltaRampUp)*start_up[unit.name, t]+(unit.MinPower+unit.DeltaRampUp)*is_on[unit.name, t]-(unit.MinPower)*is_on[unit.name, t-1])
    @constraint(model,  [unit in thermal_units, t in 1:T], power[unit.name, t-1]-power[unit.name, t]<=(-unit.DeltaRampDown)*start_down[unit.name, t]+(unit.MinPower+unit.DeltaRampDown)*is_on[unit.name, t-1]-(unit.MinPower)*is_on[unit.name, t])

    @constraint(model, thermal_cost >= sum(unit.LinearTerm*power[unit.name, t] for unit in thermal_units for t in 1:T)+sum(SHEDDING_COST*power_shedding[b,t]+CURTAILEMENT_COST*power_curtailement[b,t] for b in Buses for t in 1:T))

    @constraint(model,  [t in 1:T, b in Buses], sum(power[unit.name, t] for unit in thermal_units if unit.Bus==b)+power_shedding[b,t]-power_curtailement[b,t] + sum(flow[line.id, t] for line in Lines if line.b2==b) - sum(flow[line.id, t] for line in Lines if line.b1==b) == instance.Demandbus[b][t])

    @objective(model, Min, thermal_cost + thermal_fixed_cost)
    
    return model
end

function master_RO_problem_extended(instance; silent=true)
    """
    Initial master problem in the risk neutral 3-bin formulation
    """

    model = Model(instance.optimizer)
    if silent
        set_silent(model)
    end

    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "TimeLimit", 5)

    T= instance.TimeHorizon
    N=instance.N    
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    Numlines=length(instance.Lines)
    thermal_units=instance.Thermalunits

    @variable(model, is_on[i in 1:N, t in 0:T], Bin)
    @variable(model, start_up[i in 1:N, t in 1:T], Bin)
    @variable(model, start_down[i in 1:N, t in 1:T], Bin)

    @variable(model, gamma[i in 1:N, (a,b) in instance.Thermalunits[i].intervals]>=0)
    @constraint(model, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if a<=t<b)==is_on[unit.name,t])
    @constraint(model, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if a==t)==start_up[unit.name,t])
    @constraint(model, [unit in thermal_units, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if b==t)==start_down[unit.name,t])

    @variable(model, thermal_cost>=0)
    @variable(model, thermal_fixed_cost>=0)

    thermal_unit_commit_constraints(model, instance)
    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_units for t in 1:T))

    @objective(model, Min, thermal_cost + thermal_fixed_cost)
    
    return model
end

mutable struct oracleSP 
    model::Model
    μₘᵢₙ::Matrix{VariableRef}
    μₘₐₓ::Matrix{VariableRef}
    μꜛ::Matrix{VariableRef}
    μꜜ::Matrix{VariableRef}
    ν::Matrix{VariableRef}
    λ::Vector{VariableRef}
    network_cost::VariableRef
end

function oracle_SP_problem(instance)
    """
    Initial the subproblem in its dual form
    """

    model = Model(instance.optimizer)
    set_silent(model)
    T= instance.TimeHorizon

    N=instance.N
    thermal_units=instance.Thermalunits

    Next=instance.Next
    Buses=1:size(Next)[1]

    @variable(model, μₘᵢₙ[i in 1:N, t in 1:T]>=0)
    @variable(model, μₘₐₓ[i in 1:N, t in 1:T]>=0)
    @variable(model, μꜛ[i in 1:N, t in 1:T]>=0)
    @variable(model, μꜜ[i in 1:N, t in 1:T]>=0)
    @variable(model, ν[b in Buses, t in 1:T])
    @variable(model, λ[i in 1:N])

    @constraint(model,  [unit in thermal_units, t in 1:T-1], μₘᵢₙ[unit.name, t]-μₘₐₓ[unit.name, t]-μꜛ[unit.name,t]+μꜛ[unit.name,t+1]+μꜜ[unit.name,t]-μꜜ[unit.name,t+1]+ν[unit.Bus, t]==unit.LinearTerm)
    @constraint(model,  [unit in thermal_units], λ[unit.name]-μꜜ[unit.name, 1]+μꜛ[unit.name,1]==0)
    @constraint(model,  [unit in thermal_units], μₘᵢₙ[unit.name, T]-μₘₐₓ[unit.name, T]-μꜛ[unit.name,T]+μꜜ[unit.name,T]+ν[unit.Bus, T]==unit.LinearTerm)

    @constraint(model,  [b in Buses, t in 1:T], ν[b, t]<=SHEDDING_COST)
    @constraint(model,  [b in Buses, t in 1:T], ν[b, t]>=-CURTAILEMENT_COST)

    @variable(model, λ1[b1 in Buses, b2 in Next[b1], t in 1:T])
    @variable(model, λ2[b1 in Buses, b2 in Next[b1], t in 1:T]>=0)
    @variable(model, λ3[b1 in Buses, b2 in Next[b1], t in 1:T]>=0)

    Lines=instance.Lines
    @variable(model, network_cost)
    @constraint(model, network_cost==-sum(Lines[b1,b2].Fmax*(λ2[b1,b2,t]+λ3[b1,b2,t]) for b1 in Buses for b2 in Next[b1] for t in 1:T))
    @constraint(model, [b1 in Buses, b2 in Next[b1], t in 1:T], -ν[b1, t]+λ1[b1,b2,t]-λ2[b1,b2,t]+λ3[b1,b2,t]==0)
    @constraint(model, [b1 in Buses, t in 1:T], sum(Lines[b1,b2].B12*λ1[b1,b2,t]-Lines[b2,b1].B12*λ1[b2,b1,t] for b2 in Next[b1])==0)

    @objective(model, Max, sum(λ[unit.name]*unit.InitialPower for unit in thermal_units if unit.InitialPower!=nothing)+network_cost)

    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "TimeLimit", 15)
    set_optimizer_attribute(model, "DualReductions", 0)
    set_optimizer_attribute(model, "Presolve", 0)

    return model
    
    return oracleSP(model, μₘᵢₙ, μₘₐₓ, μꜛ, μꜜ, ν, λ, network_cost)
end