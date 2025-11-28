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

function master_RO_bin_problem_extended(instance; silent=true)
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
    N1=instance.N1
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    Numlines=length(instance.Lines)
    thermal_unitsN1=instance.Thermalunits[1:N1]

    @variable(model, is_on[i in 1:N1, t in 0:T], Bin)
    @variable(model, start_up[i in 1:N1, t in 1:T], Bin)
    @variable(model, start_down[i in 1:N1, t in 1:T], Bin)

    @variable(model, gamma[i in 1:N1, (a,b) in instance.Thermalunits[i].intervals]>=0)
    @constraint(model, [unit in thermal_unitsN1, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if a<=t<b)==is_on[unit.name,t])
    @constraint(model, [unit in thermal_unitsN1, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if a==t)==start_up[unit.name,t])
    @constraint(model, [unit in thermal_unitsN1, t in 1:T], sum(gamma[unit.name, [a,b]] for (a,b) in unit.intervals if b==t)==start_down[unit.name,t])

    @variable(model, thermal_cost>=0)
    @variable(model, thermal_fixed_cost>=0)

    thermal_unit_commit_constraintsN1(model, instance)
    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_unitsN1 for t in 1:T))

    @objective(model, Min, thermal_cost + thermal_fixed_cost)
    
    return model
end


mutable struct LagrangianRO
    model::Model
    theta::VariableRef
    upper_constraint::Union{Nothing, JuMP.ConstraintRef}
    uncertainty::Dict{Int64, VariableRef}
    dual_var_is_on::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_start_up::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_start_down::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_is_onN2::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_start_upN2::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_start_downN2::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_uncertainty::Vector{VariableRef}
    dual_demand::Matrix{VariableRef}
    vars::Vector{VariableRef}
end

mutable struct LagrangianUncertaintyRO
    model::Model
    theta::VariableRef
    upper_constraint::Union{Nothing, JuMP.ConstraintRef}
    uncertainty::Dict{Int64, VariableRef}
    dual_var_is_on::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_start_up::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_start_down::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_uncertainty::Vector{VariableRef}
    dual_demand::Matrix{VariableRef}
    vars::Vector{VariableRef}
end

mutable struct oracleRO
    model::Model
    dual_var_is_on::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_start_up::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_start_down::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_uncertainty::Vector{VariableRef}
    dual_demand::Matrix{VariableRef}
    dual_up::Matrix{VariableRef}
    dual_down::Matrix{VariableRef}
end

mutable struct oracleContinuousRO
    model::Model
    dual_var_is_on::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_start_up::Dict{Tuple{Int64, Int64}, VariableRef}
    dual_var_start_down::Dict{Tuple{Int64, Int64}, VariableRef}
    uncertainty::Vector{VariableRef}
end

mutable struct subproblemRO
    model::Model
    is_on::Dict{Tuple{Int64, Int64}, VariableRef}
    start_up::Dict{Tuple{Int64, Int64}, VariableRef}
    start_down::Dict{Tuple{Int64, Int64}, VariableRef}
    uncertainty::Vector{VariableRef}
    cstr_demand::Matrix{ConstraintRef}
end


mutable struct relaxation1
    model::Model
    muup::Matrix{ConstraintRef}
    mudown::Matrix{ConstraintRef}
    is_on::Dict{Tuple{Int64, Int64}, VariableRef}
    start_up::Dict{Tuple{Int64, Int64}, VariableRef}
    start_down::Dict{Tuple{Int64, Int64}, VariableRef}
    power::Dict{Tuple{Int64, Int64}, VariableRef}
    power_shedding::Dict{Tuple{Int64, Int64}, VariableRef}
    power_curtailement::Dict{Tuple{Int64, Int64}, VariableRef}  
    flow::Dict{Tuple{Int64, Int64}, VariableRef}
end

mutable struct relaxation2
    model::Model
    power::Dict{Tuple{Int64, Int64}, VariableRef}
end

mutable struct priceRelaxationRO
    model1::relaxation1
    model2::relaxation2
end

mutable struct twoRO
    master_pb::Model
    lagrangian::LagrangianRO
    lagrangianUncertainty::LagrangianUncertaintyRO
    subproblem::subproblemRO
    oracleContinuousRO::oracleContinuousRO
    priceRelaxation::priceRelaxationRO
end

# oracleRO(md, dual_var_is_on, dual_var_start_up, dual_var_start_down, dual_var_uncertainty, dual_demand)

function oracle_RO_problem(instance; silent=true, Γ=0)
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

    @variable(model, thermal_fuel_cost>=0)

    @variable(model, power[i in 1:N, t in 0:T]>=0)
    @variable(model, power_shedding[b in Buses, t in 1:T]>=0)
    @variable(model, power_curtailement[b in Buses, t in 1:T]>=0)

    @variable(model, uncertainty[t in 1:T], Bin)

    @constraint(model, sum(uncertainty[t] for t in 1:T) <= Γ)

    @variable(model, flow[l in 1:Numlines, t in 1:T])  
    @variable(model, θ[b in Buses, t in 1:T])  
    
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]<=line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]>=-line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]==line.B12*(θ[line.b1,t]-θ[line.b2,t]))

    @constraint(model,  [unit in thermal_units, t in 0:T], power[unit.name, t]>=unit.MinPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 0:T], power[unit.name, t]<=unit.MaxPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units; unit.InitialPower!=nothing], power[unit.name, 0]==unit.InitialPower)

    @constraint(model,  muup[i in 1:N, t in 1:T], -(power[i, t]-power[i, t-1]) >= -((-thermal_units[i].DeltaRampUp)*start_up[i, t]+(thermal_units[i].MinPower+thermal_units[i].DeltaRampUp)*is_on[i, t]-(thermal_units[i].MinPower)*is_on[i, t-1]))
    @constraint(model,  mudown[i in 1:N, t in 1:T], -(power[i, t-1]-power[i, t]) >= -((-thermal_units[i].DeltaRampDown)*start_down[i, t]+(thermal_units[i].MinPower+thermal_units[i].DeltaRampDown)*is_on[i, t-1]-(thermal_units[i].MinPower)*is_on[i, t]))

    @constraint(model, thermal_fuel_cost >= sum(unit.LinearTerm*power[unit.name, t] for unit in thermal_units for t in 1:T)+sum(SHEDDING_COST*power_shedding[b,t]+CURTAILEMENT_COST*power_curtailement[b,t] for b in Buses for t in 1:T))

    @constraint(model,  demand[t in 1:T, b in Buses], sum(power[unit.name, t] for unit in thermal_units if unit.Bus==b)+power_shedding[b,t]-power_curtailement[b,t] + sum(flow[line.id, t] for line in Lines if line.b2==b) - sum(flow[line.id, t] for line in Lines if line.b1==b) == instance.Demandbus[b][t]*(1+1.96*0.025*uncertainty[t]))

    @objective(model, Min, thermal_fuel_cost)

    undo_relax = JuMP.relax_integrality(model)
    new_model, _ = safe_copy(model)
    undo_relax()

    copy_is_on = Dict()
    copy_start_up = Dict()
    copy_start_down = Dict()
    copy_uncertainty = Dict()
    for i in 1:N
        for t in 0:T
            copy_is_on[i,t] = JuMP.variable_by_name(new_model, string("is_on[",i,",",t,"]"))
        end
    end
    for i in 1:N
        for t in 1:T
            copy_start_up[i,t] = JuMP.variable_by_name(new_model, string("start_up[",i,",",t,"]"))
            copy_start_down[i,t] = JuMP.variable_by_name(new_model, string("start_down[",i,",",t,"]"))
        end
    end
    for t in 1:T
        copy_uncertainty[t] = JuMP.variable_by_name(new_model, string("uncertainty[",t,"]"))
    end

    @constraint(new_model, fix_is_on[i in 1:N, t in 0:T], copy_is_on[i, t] == 0.0)
    @constraint(new_model, fix_start_up[i in 1:N, t in 1:T], copy_start_up[i, t] == 0.0)
    @constraint(new_model, fix_start_down[i in 1:N, t in 1:T], copy_start_down[i, t] == 0.0)
    @constraint(new_model, fix_uncertainty[t in 1:T], copy_uncertainty[t] == 0.0)

    md=dualize(new_model; consider_constrained_variables=false, dual_names = DualNames("dual_var_", "dual_con_"))

    hexpr = JuMP.objective_function(md)
    theta = @variable(
        md,
        base_name = "theta",
    )
    @constraint(md, theta <= hexpr)

    dual_var_is_on = Dict((i,t) => md[Symbol("dual_var_fix_is_on[$i,$t]")] for (i,t) in keys(copy_is_on))
    dual_var_start_up = Dict((i,t) => md[Symbol("dual_var_fix_start_up[$i,$t]")] for (i,t) in keys(copy_start_up))
    dual_var_start_down = Dict((i,t) => md[Symbol("dual_var_fix_start_down[$i,$t]")] for (i,t) in keys(copy_start_down))
    dual_var_uncertainty = [md[Symbol("dual_var_fix_uncertainty[$t]")] for t in 1:T]
    dual_demand = [md[Symbol("dual_var_demand[$t,$b]")] for t in 1:T, b in Buses]
    dual_up = [md[Symbol("dual_var_muup[$i,$t]")] for i in 1:N, t in 1:T]
    dual_down = [md[Symbol("dual_var_mudown[$i,$t]")] for i in 1:N, t in 1:T]

    indexes = keys(dual_var_uncertainty)
    @variable(md, δ[indexes])
    @variable(md, uncertainty[indexes], Bin)
    @constraint(md, sum(uncertainty[t] for t in 1:T) <= Γ)
    max_D = maximum([sum(d[t]*0.025*1.96 for d in instance.Demandbus) for t in 1:T])
    @constraint(md, [k in indexes], δ[k] <= SHEDDING_COST * max_D * uncertainty[k])
    @constraint(md, [k in indexes], δ[k] >= -CURTAILEMENT_COST * max_D * uncertainty[k])
    @constraint(md, [k in indexes], δ[k] <= dual_var_uncertainty[k] + CURTAILEMENT_COST * max_D * (1 - uncertainty[k]))
    @constraint(md, [k in indexes], δ[k] >= dual_var_uncertainty[k] - SHEDDING_COST * max_D * (1 - uncertainty[k]))
    JuMP.set_objective_function(
        md,
            @expression(md, theta + sum(δ)),
    )

    set_optimizer(md, instance.optimizer)    
    set_optimizer_attribute(md, "Threads", 1)
    set_optimizer_attribute(md, "TimeLimit", 5)
    if silent
        set_silent(md)
    end
    
    return oracleRO(md, dual_var_is_on, dual_var_start_up, dual_var_start_down, dual_var_uncertainty, dual_demand, dual_up, dual_down)
end

function Lagrangian_RO_problem(instance; silent=true, Γ=0)
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
    N1 = instance.N1
    N2 = N - N1
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    Numlines=length(instance.Lines)
    thermal_units=instance.Thermalunits
    thermal_units_2=values(instance.Thermalunits)[N1+1:N]

    @variable(model, is_on[i in 1:N, t in 0:T], Bin)
    @variable(model, start_up[i in 1:N, t in 1:T], Bin)
    @variable(model, start_down[i in 1:N, t in 1:T], Bin)

    @variable(model, thermal_fixed_cost>=0)

    thermal_unit_commit_constraints(model, instance)
    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_units for t in 1:T if unit.name > N1))

    @variable(model, thermal_fuel_cost>=0)

    @variable(model, power[i in 1:N, t in 0:T]>=0)
    @variable(model, power_shedding[b in Buses, t in 1:T]>=0)
    @variable(model, power_curtailement[b in Buses, t in 1:T]>=0)

    @variable(model, uncertainty[t in 1:T], Bin)

    @constraint(model, sum(uncertainty[t] for t in 1:T) <= Γ)

    @variable(model, flow[l in 1:Numlines, t in 1:T])  
    @variable(model, θ[b in Buses, t in 1:T])  
    
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]<=line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]>=-line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]==line.B12*(θ[line.b1,t]-θ[line.b2,t]))

    @constraint(model,  [unit in thermal_units, t in 0:T], power[unit.name, t]>=unit.MinPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 0:T], power[unit.name, t]<=unit.MaxPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units; unit.InitialPower!=nothing], power[unit.name, 0]==unit.InitialPower)

    @constraint(model,  muup[i in 1:N, t in 1:T], -(power[i, t]-power[i, t-1]) >= -((-thermal_units[i].DeltaRampUp)*start_up[i, t]+(thermal_units[i].MinPower+thermal_units[i].DeltaRampUp)*is_on[i, t]-(thermal_units[i].MinPower)*is_on[i, t-1]))
    @constraint(model,  mudown[i in 1:N, t in 1:T], -(power[i, t-1]-power[i, t]) >= -((-thermal_units[i].DeltaRampDown)*start_down[i, t]+(thermal_units[i].MinPower+thermal_units[i].DeltaRampDown)*is_on[i, t-1]-(thermal_units[i].MinPower)*is_on[i, t]))

    @constraint(model, thermal_fuel_cost >= thermal_fixed_cost + sum(unit.LinearTerm*power[unit.name, t] for unit in thermal_units for t in 1:T)+sum(SHEDDING_COST*power_shedding[b,t]+CURTAILEMENT_COST*power_curtailement[b,t] for b in Buses for t in 1:T))

    @constraint(model,  demand[t in 1:T, b in Buses], sum(power[unit.name, t] for unit in thermal_units if unit.Bus==b)+power_shedding[b,t]-power_curtailement[b,t] + sum(flow[line.id, t] for line in Lines if line.b2==b) - sum(flow[line.id, t] for line in Lines if line.b1==b) == instance.Demandbus[b][t]*(1+1.96*0.025*uncertainty[t]))

    @objective(model, Min, thermal_fuel_cost)

    undo_relax = JuMP.relax_integrality(model)
    new_model, _ = safe_copy(model)
    undo_relax()

    copy_is_on = Dict()
    copy_start_up = Dict()
    copy_start_down = Dict()
    copy_uncertainty = Dict()
    for i in 1:N1
        for t in 0:T
            copy_is_on[i,t] = JuMP.variable_by_name(new_model, string("is_on[",i,",",t,"]"))
        end
    end
    for i in 1:N1
        for t in 1:T
            copy_start_up[i,t] = JuMP.variable_by_name(new_model, string("start_up[",i,",",t,"]"))
            copy_start_down[i,t] = JuMP.variable_by_name(new_model, string("start_down[",i,",",t,"]"))
        end
    end

    copy_is_onN2 = Dict()
    copy_start_upN2 = Dict()
    copy_start_downN2 = Dict()
    for k in 1:N2
        i = k + N1
        for t in 0:T
            copy_is_onN2[k,t] = JuMP.variable_by_name(new_model, string("is_on[",i,",",t,"]"))
        end
    end
    for k in 1:N2
        i = k + N1
        for t in 1:T
            copy_start_upN2[k,t] = JuMP.variable_by_name(new_model, string("start_up[",i,",",t,"]"))
            copy_start_downN2[k,t] = JuMP.variable_by_name(new_model, string("start_down[",i,",",t,"]"))
        end
    end



    for t in 1:T
        copy_uncertainty[t] = JuMP.variable_by_name(new_model, string("uncertainty[",t,"]"))
    end

    @constraint(new_model, fix_is_on[i in 1:N1, t in 0:T], copy_is_on[i, t] == 0.0)
    @constraint(new_model, fix_start_up[i in 1:N1, t in 1:T], copy_start_up[i, t] == 0.0)
    @constraint(new_model, fix_start_down[i in 1:N1, t in 1:T], copy_start_down[i, t] == 0.0)

    @constraint(new_model, fix_uncertainty[t in 1:T], copy_uncertainty[t] == 0.0)

    @constraint(new_model, fix_is_onN2[k in 1:N2, t in 0:T], copy_is_onN2[k, t] == 0.0)
    @constraint(new_model, fix_start_upN2[k in 1:N2, t in 1:T], copy_start_upN2[k, t] == 0.0)
    @constraint(new_model, fix_start_downN2[k in 1:N2, t in 1:T], copy_start_downN2[k, t] == 0.0)

    md=dualize(new_model; consider_constrained_variables=false, dual_names = DualNames("dual_var_", "dual_con_"))

    hexpr = JuMP.objective_function(md)
    theta = @variable(
        md,
        base_name = "theta",
    )
    upper_cstr = @constraint(md, theta <= hexpr)

    dual_var_is_on = Dict((i,t) => md[Symbol("dual_var_fix_is_on[$i,$t]")] for (i,t) in keys(copy_is_on))
    dual_var_start_up = Dict((i,t) => md[Symbol("dual_var_fix_start_up[$i,$t]")] for (i,t) in keys(copy_start_up))
    dual_var_start_down = Dict((i,t) => md[Symbol("dual_var_fix_start_down[$i,$t]")] for (i,t) in keys(copy_start_down))

    dual_var_is_onN2 = Dict((k,t) => md[Symbol("dual_var_fix_is_onN2[$k,$t]")] for (k,t) in keys(copy_is_onN2))
    dual_var_start_upN2 = Dict((k,t) => md[Symbol("dual_var_fix_start_upN2[$k,$t]")] for (k,t) in keys(copy_start_upN2))
    dual_var_start_downN2 = Dict((k,t) => md[Symbol("dual_var_fix_start_downN2[$k,$t]")] for (k,t) in keys(copy_start_downN2))

    dual_demand = [md[Symbol("dual_var_demand[$t,$b]")] for t in 1:T, b in Buses]

    dual_var_uncertainty = [md[Symbol("dual_var_fix_uncertainty[$t]")] for t in 1:T]

    indexes = keys(dual_var_uncertainty)
    @variable(md, δ[indexes])
    @variable(md, uncertainty[indexes], Bin)
    ξ = Dict(t => uncertainty[t] for t in 1:T)
    @constraint(md, sum(uncertainty[t] for t in 1:T) <= Γ)
    max_D = [sum(d[t]*0.025*1.96 for d in instance.Demandbus) for t in 1:T]
    @constraint(md, [k in indexes], δ[k] <= SHEDDING_COST * max_D[k] * uncertainty[k])
    @constraint(md, [k in indexes], δ[k] >= -CURTAILEMENT_COST * max_D[k] * uncertainty[k])
    @constraint(md, [k in indexes], δ[k] <= dual_var_uncertainty[k] + CURTAILEMENT_COST * max_D[k] * (1 - uncertainty[k]))
    @constraint(md, [k in indexes], δ[k] >= dual_var_uncertainty[k] - SHEDDING_COST * max_D[k] * (1 - uncertainty[k]))
    JuMP.set_objective_function(
        md,
            @expression(md, theta + sum(δ)),
    )

    set_optimizer(md, instance.optimizer)    
    set_optimizer_attribute(md, "Threads", 1)
    set_optimizer_attribute(md, "TimeLimit", 20)
    if silent
        set_silent(md)
    end

    vars = [v for v in JuMP.all_variables(md) if JuMP.name(v) != "theta"]
    
    return LagrangianRO(md, theta, upper_cstr, ξ, dual_var_is_on, dual_var_start_up, dual_var_start_down, dual_var_is_onN2, dual_var_start_upN2, dual_var_start_downN2, dual_var_uncertainty, dual_demand, vars)
end

function Lagrangian_uncertainty_RO_problem(instance; silent=true, Γ=0)
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
    N1 = instance.N1    
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    Numlines=length(instance.Lines)
    thermal_units=instance.Thermalunits

    @variable(model, is_on[i in 1:N, t in 0:T], Bin)
    @variable(model, start_up[i in 1:N, t in 1:T], Bin)
    @variable(model, start_down[i in 1:N, t in 1:T], Bin)

    @variable(model, thermal_fixed_cost>=0)

    thermal_unit_commit_constraints(model, instance)
    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_units for t in 1:T if unit.name > N1))

    @variable(model, thermal_fuel_cost>=0)

    @variable(model, power[i in 1:N, t in 0:T]>=0)
    @variable(model, power_shedding[b in Buses, t in 1:T]>=0)
    @variable(model, power_curtailement[b in Buses, t in 1:T]>=0)

    @variable(model, uncertainty[t in 1:T], Bin)

    @constraint(model, sum(uncertainty[t] for t in 1:T) <= Γ)

    @variable(model, flow[l in 1:Numlines, t in 1:T])  
    @variable(model, θ[b in Buses, t in 1:T])  
    
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]<=line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]>=-line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]==line.B12*(θ[line.b1,t]-θ[line.b2,t]))

    @constraint(model,  [unit in thermal_units, t in 0:T], power[unit.name, t]>=unit.MinPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 0:T], power[unit.name, t]<=unit.MaxPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units; unit.InitialPower!=nothing], power[unit.name, 0]==unit.InitialPower)

    @constraint(model,  muup[i in 1:N, t in 1:T], -(power[i, t]-power[i, t-1]) >= -((-thermal_units[i].DeltaRampUp)*start_up[i, t]+(thermal_units[i].MinPower+thermal_units[i].DeltaRampUp)*is_on[i, t]-(thermal_units[i].MinPower)*is_on[i, t-1]))
    @constraint(model,  mudown[i in 1:N, t in 1:T], -(power[i, t-1]-power[i, t]) >= -((-thermal_units[i].DeltaRampDown)*start_down[i, t]+(thermal_units[i].MinPower+thermal_units[i].DeltaRampDown)*is_on[i, t-1]-(thermal_units[i].MinPower)*is_on[i, t]))

    @constraint(model, thermal_fuel_cost >= thermal_fixed_cost + sum(unit.LinearTerm*power[unit.name, t] for unit in thermal_units for t in 1:T)+sum(SHEDDING_COST*power_shedding[b,t]+CURTAILEMENT_COST*power_curtailement[b,t] for b in Buses for t in 1:T))

    @constraint(model,  demand[t in 1:T, b in Buses], sum(power[unit.name, t] for unit in thermal_units if unit.Bus==b)+power_shedding[b,t]-power_curtailement[b,t] + sum(flow[line.id, t] for line in Lines if line.b2==b) - sum(flow[line.id, t] for line in Lines if line.b1==b) == instance.Demandbus[b][t]*(1+1.96*0.025*uncertainty[t]))

    @objective(model, Min, thermal_fuel_cost)

    undo_relax = JuMP.relax_integrality(model)
    new_model, _ = safe_copy(model)
    undo_relax()

    copy_is_on = Dict()
    copy_start_up = Dict()
    copy_start_down = Dict()
    copy_uncertainty = Dict()
    for i in 1:N
        for t in 0:T
            copy_is_on[i,t] = JuMP.variable_by_name(new_model, string("is_on[",i,",",t,"]"))
        end
    end
    for i in 1:N
        for t in 1:T
            copy_start_up[i,t] = JuMP.variable_by_name(new_model, string("start_up[",i,",",t,"]"))
            copy_start_down[i,t] = JuMP.variable_by_name(new_model, string("start_down[",i,",",t,"]"))
        end
    end

    for t in 1:T
        copy_uncertainty[t] = JuMP.variable_by_name(new_model, string("uncertainty[",t,"]"))
    end

    @constraint(new_model, fix_is_on[i in 1:N, t in 0:T], copy_is_on[i, t] == 0.0)
    @constraint(new_model, fix_start_up[i in 1:N, t in 1:T], copy_start_up[i, t] == 0.0)
    @constraint(new_model, fix_start_down[i in 1:N, t in 1:T], copy_start_down[i, t] == 0.0)

    @constraint(new_model, fix_uncertainty[t in 1:T], copy_uncertainty[t] == 0.0)

    md=dualize(new_model; consider_constrained_variables=false, dual_names = DualNames("dual_var_", "dual_con_"))

    hexpr = JuMP.objective_function(md)
    theta = @variable(
        md,
        base_name = "theta",
    )
    upper_cstr = @constraint(md, theta <= hexpr)

    dual_var_is_on = Dict((i,t) => md[Symbol("dual_var_fix_is_on[$i,$t]")] for (i,t) in keys(copy_is_on))
    dual_var_start_up = Dict((i,t) => md[Symbol("dual_var_fix_start_up[$i,$t]")] for (i,t) in keys(copy_start_up))
    dual_var_start_down = Dict((i,t) => md[Symbol("dual_var_fix_start_down[$i,$t]")] for (i,t) in keys(copy_start_down))

    dual_demand = [md[Symbol("dual_var_demand[$t,$b]")] for t in 1:T, b in Buses]

    dual_var_uncertainty = [md[Symbol("dual_var_fix_uncertainty[$t]")] for t in 1:T]

    indexes = keys(dual_var_uncertainty)
    @variable(md, δ[indexes])
    @variable(md, uncertainty[indexes], Bin)
    ξ = Dict(t => uncertainty[t] for t in 1:T)
    @constraint(md, sum(uncertainty[t] for t in 1:T) <= Γ)
    max_D = [sum(d[t]*0.025*1.96 for d in instance.Demandbus) for t in 1:T]
    @constraint(md, [k in indexes], δ[k] <= SHEDDING_COST * max_D[k] * uncertainty[k])
    @constraint(md, [k in indexes], δ[k] >= -CURTAILEMENT_COST * max_D[k] * uncertainty[k])
    @constraint(md, [k in indexes], δ[k] <= dual_var_uncertainty[k] + CURTAILEMENT_COST * max_D[k] * (1 - uncertainty[k]))
    @constraint(md, [k in indexes], δ[k] >= dual_var_uncertainty[k] - SHEDDING_COST * max_D[k] * (1 - uncertainty[k]))
    JuMP.set_objective_function(
        md,
            @expression(md, theta + sum(δ)),
    )

    set_optimizer(md, instance.optimizer)    
    set_optimizer_attribute(md, "Threads", 1)
    set_optimizer_attribute(md, "TimeLimit", 60)
    if silent
        set_silent(md)
    end

    vars = [v for v in JuMP.all_variables(md) if JuMP.name(v) != "theta"]
    
    return LagrangianUncertaintyRO(md, theta, upper_cstr, ξ, dual_var_is_on, dual_var_start_up, dual_var_start_down, dual_var_uncertainty, dual_demand, vars)
end

function subproblemRO(instance, Γ)
    model = Model(instance.optimizer)

    set_silent(model)

    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "TimeLimit", 5)

    T= instance.TimeHorizon
    N=instance.N    
    N1 = instance.N1
    N2 = N - N1
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    Numlines=length(instance.Lines)
    thermal_units=instance.Thermalunits

    @variable(model, is_on[i in 1:N, t in 0:T]>=0)
    @variable(model, start_up[i in 1:N, t in 1:T]>=0)
    @variable(model, start_down[i in 1:N, t in 1:T]>=0)

    @variable(model, thermal_fuel_cost>=0)

    @variable(model, power[i in 1:N, t in 0:T]>=0)
    @variable(model, power_shedding[b in Buses, t in 1:T]>=0)
    @variable(model, power_curtailement[b in Buses, t in 1:T]>=0)

    # @variable(model, uncertainty[t in 1:T]>=0)
    @variable(model, uncertainty[t in 1:T], Bin)


    @constraint(model, [i in 1:N, t in 1:T], is_on[i,t] <= 1)
    @constraint(model, [i in 1:N, t in 1:T], start_up[i,t] <= 1)
    @constraint(model, [i in 1:N, t in 1:T], start_down[i,t] <= 1)
    @constraint(model, [t in 1:T], uncertainty[t] <= 1)

    @constraint(model, sum(uncertainty[t] for t in 1:T) <= Γ)

    for i in 1:N
        for t in 0:T
            JuMP.set_binary(is_on[i, t])
        end
        for t in 1:T
            JuMP.set_binary(start_up[i, t])
            JuMP.set_binary(start_down[i, t])
        end
    end

    @variable(model, thermal_fixed_cost>=0)

    thermal_unit_commit_constraints(model, instance)
    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_units for t in 1:T if unit.name > N1))

    @variable(model, flow[l in 1:Numlines, t in 1:T])  
    @variable(model, θ[b in Buses, t in 1:T])  
    
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]<=line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]>=-line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]==line.B12*(θ[line.b1,t]-θ[line.b2,t]))

    @constraint(model,  [unit in thermal_units, t in 0:T], power[unit.name, t]>=unit.MinPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 0:T], power[unit.name, t]<=unit.MaxPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units; unit.InitialPower!=nothing], power[unit.name, 0]==unit.InitialPower)

    @constraint(model,  muup[i in 1:N, t in 1:T], -(power[i, t]-power[i, t-1]) >= -((-thermal_units[i].DeltaRampUp)*start_up[i, t]+(thermal_units[i].MinPower+thermal_units[i].DeltaRampUp)*is_on[i, t]-(thermal_units[i].MinPower)*is_on[i, t-1]))
    @constraint(model,  mudown[i in 1:N, t in 1:T], -(power[i, t-1]-power[i, t]) >= -((-thermal_units[i].DeltaRampDown)*start_down[i, t]+(thermal_units[i].MinPower+thermal_units[i].DeltaRampDown)*is_on[i, t-1]-(thermal_units[i].MinPower)*is_on[i, t]))

    @constraint(model, thermal_fuel_cost >= thermal_fixed_cost + sum(unit.LinearTerm*power[unit.name, t] for unit in thermal_units for t in 1:T)+sum(SHEDDING_COST*power_shedding[b,t]+CURTAILEMENT_COST*power_curtailement[b,t] for b in Buses for t in 1:T))

    @constraint(model,  demand[t in 1:T, b in Buses], sum(power[unit.name, t] for unit in thermal_units if unit.Bus==b)+power_shedding[b,t]-power_curtailement[b,t] + sum(flow[line.id, t] for line in Lines if line.b2==b) - sum(flow[line.id, t] for line in Lines if line.b1==b) == instance.Demandbus[b][t]*(1+1.96*0.025*uncertainty[t]))

    cstr_demand = Matrix{ConstraintRef}(undef, T, length(Buses))
    for t in 1:T
        for b in Buses
            cstr_demand[t,b] = demand[t,b]
        end
    end

    @objective(model, Min, thermal_fuel_cost)

    return subproblemRO(model, Dict((i,t) => JuMP.variable_by_name(model, string("is_on[",i,",",t,"]")) for i in 1:N for t in 0:T),
                        Dict((i,t) => JuMP.variable_by_name(model, string("start_up[",i,",",t,"]")) for i in 1:N for t in 1:T),
                        Dict((i,t) => JuMP.variable_by_name(model, string("start_down[",i,",",t,"]")) for i in 1:N for t in 1:T),
                        [JuMP.variable_by_name(model, string("uncertainty[",t,"]")) for t in 1:T], cstr_demand)
end

function oracle_Continuous_RO_problem(instance; silent=true, Γ=0)
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
    N1 = instance.N1
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    Numlines=length(instance.Lines)
    thermal_units=instance.Thermalunits

    @variable(model, is_on[i in 1:N, t in 0:T], Bin)
    @variable(model, start_up[i in 1:N, t in 1:T], Bin)
    @variable(model, start_down[i in 1:N, t in 1:T], Bin)

    @variable(model, thermal_fixed_cost>=0)

    thermal_unit_commit_constraints(model, instance)
    @constraint(model, thermal_fixed_cost>=sum(unit.ConstTerm*is_on[unit.name, t]+unit.StartUpCost*start_up[unit.name, t]+unit.StartDownCost*start_down[unit.name, t] for unit in thermal_units for t in 1:T if unit.name > N1))

    @variable(model, thermal_fuel_cost>=0)

    @variable(model, power[i in 1:N, t in 0:T]>=0)
    @variable(model, power_shedding[b in Buses, t in 1:T]>=0)
    @variable(model, power_curtailement[b in Buses, t in 1:T]>=0)

    @variable(model, uncertainty[t in 1:T], Bin)

    @constraint(model, sum(uncertainty[t] for t in 1:T) <= Γ)

    @variable(model, flow[l in 1:Numlines, t in 1:T])  
    @variable(model, θ[b in Buses, t in 1:T])  
    
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]<=line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]>=-line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]==line.B12*(θ[line.b1,t]-θ[line.b2,t]))

    @constraint(model,  [unit in thermal_units, t in 0:T], power[unit.name, t]>=unit.MinPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units, t in 0:T], power[unit.name, t]<=unit.MaxPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units; unit.InitialPower!=nothing], power[unit.name, 0]==unit.InitialPower)

    @constraint(model,  muup[i in 1:N, t in 1:T], -(power[i, t]-power[i, t-1]) >= -((-thermal_units[i].DeltaRampUp)*start_up[i, t]+(thermal_units[i].MinPower+thermal_units[i].DeltaRampUp)*is_on[i, t]-(thermal_units[i].MinPower)*is_on[i, t-1]))
    @constraint(model,  mudown[i in 1:N, t in 1:T], -(power[i, t-1]-power[i, t]) >= -((-thermal_units[i].DeltaRampDown)*start_down[i, t]+(thermal_units[i].MinPower+thermal_units[i].DeltaRampDown)*is_on[i, t-1]-(thermal_units[i].MinPower)*is_on[i, t]))

    @constraint(model, thermal_fuel_cost >= thermal_fixed_cost + sum(unit.LinearTerm*power[unit.name, t] for unit in thermal_units for t in 1:T)+sum(SHEDDING_COST*power_shedding[b,t]+CURTAILEMENT_COST*power_curtailement[b,t] for b in Buses for t in 1:T))

    @constraint(model,  demand[t in 1:T, b in Buses], sum(power[unit.name, t] for unit in thermal_units if unit.Bus==b)+power_shedding[b,t]-power_curtailement[b,t] + sum(flow[line.id, t] for line in Lines if line.b2==b) - sum(flow[line.id, t] for line in Lines if line.b1==b) == instance.Demandbus[b][t]*(1+1.96*0.025*uncertainty[t]))

    @objective(model, Min, thermal_fuel_cost)

    undo_relax = JuMP.relax_integrality(model)
    new_model, _ = safe_copy(model)
    undo_relax()

    copy_is_on = Dict()
    copy_start_up = Dict()
    copy_start_down = Dict()
    copy_uncertainty = Dict()
    for i in 1:N1
        for t in 0:T
            copy_is_on[i,t] = JuMP.variable_by_name(new_model, string("is_on[",i,",",t,"]"))
        end
    end
    for i in 1:N1
        for t in 1:T
            copy_start_up[i,t] = JuMP.variable_by_name(new_model, string("start_up[",i,",",t,"]"))
            copy_start_down[i,t] = JuMP.variable_by_name(new_model, string("start_down[",i,",",t,"]"))
        end
    end
    for t in 1:T
        copy_uncertainty[t] = JuMP.variable_by_name(new_model, string("uncertainty[",t,"]"))
    end

    @constraint(new_model, fix_is_on[i in 1:N1, t in 0:T], copy_is_on[i, t] == 0.0)
    @constraint(new_model, fix_start_up[i in 1:N1, t in 1:T], copy_start_up[i, t] == 0.0)
    @constraint(new_model, fix_start_down[i in 1:N1, t in 1:T], copy_start_down[i, t] == 0.0)
    @constraint(new_model, fix_uncertainty[t in 1:T], copy_uncertainty[t] == 0.0)

    md=dualize(new_model; consider_constrained_variables=false, dual_names = DualNames("dual_var_", "dual_con_"))

    hexpr = JuMP.objective_function(md)
    theta = @variable(
        md,
        base_name = "theta",
    )
    @constraint(md, theta <= hexpr)

    dual_var_is_on = Dict((i,t) => md[Symbol("dual_var_fix_is_on[$i,$t]")] for (i,t) in keys(copy_is_on))
    dual_var_start_up = Dict((i,t) => md[Symbol("dual_var_fix_start_up[$i,$t]")] for (i,t) in keys(copy_start_up))
    dual_var_start_down = Dict((i,t) => md[Symbol("dual_var_fix_start_down[$i,$t]")] for (i,t) in keys(copy_start_down))
    dual_var_uncertainty = [md[Symbol("dual_var_fix_uncertainty[$t]")] for t in 1:T]
    
    indexes = keys(dual_var_uncertainty)
    @variable(md, δ[indexes])
    @variable(md, uncertainty[indexes], Bin)
    ξ = [uncertainty[t] for t in 1:T]
    @constraint(md, sum(uncertainty[t] for t in 1:T) <= Γ)
    max_D = maximum([sum(d[t]*0.025*1.96 for d in instance.Demandbus) for t in 1:T])
    @constraint(md, [k in indexes], δ[k] <= SHEDDING_COST * max_D * uncertainty[k])
    @constraint(md, [k in indexes], δ[k] >= -CURTAILEMENT_COST * max_D * uncertainty[k])
    @constraint(md, [k in indexes], δ[k] <= dual_var_uncertainty[k] + CURTAILEMENT_COST * max_D * (1 - uncertainty[k]))
    @constraint(md, [k in indexes], δ[k] >= dual_var_uncertainty[k] - SHEDDING_COST * max_D * (1 - uncertainty[k]))
    JuMP.set_objective_function(
        md,
            @expression(md, theta + sum(δ)),
    )

    set_optimizer(md, instance.optimizer)    
    set_optimizer_attribute(md, "Threads", 1)
    set_optimizer_attribute(md, "TimeLimit", 5)
    if silent
        set_silent(md)
    end
    
    return oracleContinuousRO(md, dual_var_is_on, dual_var_start_up, dual_var_start_down, ξ)
end

function evaluate_solution(instance, solution, Γ=0; silent=true)
   model = Model(instance.optimizer)
    if silent
        set_silent(model)
    end

    set_optimizer_attribute(model, "Threads", 1)

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

    @constraint(model, [i in 1:N, t in 0:T], is_on[i,t]==solution[1][i,t])
    @constraint(model, [i in 1:N, t in 1:T], start_up[i,t]==solution[2][i,t])
    @constraint(model, [i in 1:N, t in 1:T], start_down[i,t]==solution[3][i,t])

    @variable(model, thermal_fuel_cost>=0)

    @variable(model, power[i in 1:N, t in 0:T]>=0)
    @variable(model, power_shedding[b in Buses, t in 1:T]>=0)
    @variable(model, power_curtailement[b in Buses, t in 1:T]>=0)

    @variable(model, uncertainty[t in 1:T], Bin)

    @constraint(model, sum(uncertainty[t] for t in 1:T) <= Γ)

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

    @constraint(model, thermal_fuel_cost >= sum(unit.LinearTerm*power[unit.name, t] for unit in thermal_units for t in 1:T)+sum(SHEDDING_COST*power_shedding[b,t]+CURTAILEMENT_COST*power_curtailement[b,t] for b in Buses for t in 1:T))

    @constraint(model,  [t in 1:T, b in Buses], sum(power[unit.name, t] for unit in thermal_units if unit.Bus==b)+power_shedding[b,t]-power_curtailement[b,t] + sum(flow[line.id, t] for line in Lines if line.b2==b) - sum(flow[line.id, t] for line in Lines if line.b1==b) == instance.Demandbus[b][t]*(1+1.96*0.025*uncertainty[t]))

    @objective(model, Min, thermal_fuel_cost)
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


function decompose_problem_N1(instance)
    """
    Initial master problem in the risk neutral 3-bin formulation
    """

    model = Model(instance.optimizer)
    set_silent(model)

    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "TimeLimit", 5)

    T= instance.TimeHorizon
    N=instance.N        
    N1 = instance.N1
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    Numlines=length(instance.Lines)
    thermal_units=instance.Thermalunits
    thermal_units_N1 = [unit for unit in thermal_units if unit.name <= N1]

    @variable(model, is_on[i in 1:N1, t in 0:T])
    @variable(model, start_up[i in 1:N1, t in 1:T])
    @variable(model, start_down[i in 1:N1, t in 1:T])

    thermal_unit_commit_constraintsN1(model, instance)

    @variable(model, thermal_fuel_cost>=0)

    @variable(model, power[i in 1:N1, t in 0:T]>=0)
    @variable(model, power_shedding[b in Buses, t in 1:T]>=0)
    @variable(model, power_curtailement[b in Buses, t in 1:T]>=0)

    @variable(model, flow[l in 1:Numlines, t in 1:T])  
    @variable(model, θ[b in Buses, t in 1:T])  
    
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]<=line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]>=-line.Fmax)
    @constraint(model, [line in Lines, t in 1:T], flow[line.id,t]==line.B12*(θ[line.b1,t]-θ[line.b2,t]))

    @constraint(model,  [unit in thermal_units_N1, t in 0:T], power[unit.name, t]>=unit.MinPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units_N1, t in 0:T], power[unit.name, t]<=unit.MaxPower*is_on[unit.name, t])
    @constraint(model,  [unit in thermal_units_N1; unit.InitialPower!=nothing], power[unit.name, 0]==unit.InitialPower)

    @constraint(model,  muup[i in 1:N1, t in 1:T], -(power[i, t]-power[i, t-1]) >= -((-thermal_units[i].DeltaRampUp)*start_up[i, t]+(thermal_units[i].MinPower+thermal_units[i].DeltaRampUp)*is_on[i, t]-(thermal_units[i].MinPower)*is_on[i, t-1]))
    @constraint(model,  mudown[i in 1:N1, t in 1:T], -(power[i, t-1]-power[i, t]) >= -((-thermal_units[i].DeltaRampDown)*start_down[i, t]+(thermal_units[i].MinPower+thermal_units[i].DeltaRampDown)*is_on[i, t-1]-(thermal_units[i].MinPower)*is_on[i, t]))

    @constraint(model, thermal_fuel_cost >= sum(unit.LinearTerm*power[unit.name, t] for unit in thermal_units_N1 for t in 1:T)+sum(SHEDDING_COST*power_shedding[b,t]+CURTAILEMENT_COST*power_curtailement[b,t] for b in Buses for t in 1:T))

    @objective(model, Min, thermal_fuel_cost)

    cstr_muup = Matrix{ConstraintRef}(undef, N1, T)
    cstr_mudown = Matrix{ConstraintRef}(undef, N1, T)
    for i in 1:N1
        for t in 1:T
            cstr_muup[i,t] = muup[i,t]
            cstr_mudown[i,t] = mudown[i,t]
        end
    end

    is_on1 = Dict((i,t) => JuMP.variable_by_name(model, string("is_on[",i,",",t,"]")) for i in 1:N1 for t in 0:T)
    start_up1 = Dict((i,t) => JuMP.variable_by_name(model, string("start_up[",i,",",t,"]")) for i in 1:N1 for t in 1:T)
    start_down1 = Dict((i,t) => JuMP.variable_by_name(model, string("start_down[",i,",",t,"]")) for i in 1:N1 for t in 1:T)
    power1 = Dict((i,t) => JuMP.variable_by_name(model, string("power[",i,",",t,"]")) for i in 1:N1 for t in 0:T)
    power_shedding1 = Dict((b,t) => JuMP.variable_by_name(model, string("power_shedding[",b,",",t,"]")) for b in Buses for t in 1:T)
    power_curtailement1 = Dict((b,t) => JuMP.variable_by_name(model, string("power_curtailement[",b,",",t,"]")) for b in Buses for t in 1:T)

    flow1 = Dict((l,t) => JuMP.variable_by_name(model, string("flow[",l,",",t,"]")) for l in 1:Numlines for t in 1:T)
    
    return relaxation1(model, cstr_muup, cstr_mudown, is_on1, start_up1, start_down1, power1, power_shedding1, power_curtailement1, flow1)
end

function decompose_problem_N2(instance)
    """
    Initial master problem in the risk neutral 3-bin formulation
    """

    model = Model(instance.optimizer)
    set_silent(model)

    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "TimeLimit", 5)

    T= instance.TimeHorizon
    N=instance.N        
    N1 = instance.N1
    N2 = N - N1
    thermal_units=instance.Thermalunits
    thermal_units_N2 = [unit for unit in thermal_units if unit.name > N1]

    @variable(model, is_on[i in 1:N2, t in 0:T], Bin)
    @variable(model, start_up[i in 1:N2, t in 1:T], Bin)
    @variable(model, start_down[i in 1:N2, t in 1:T], Bin)

    @variable(model, thermal_fixed_cost>=0)

    thermal_unit_commit_constraintsN2(model, instance)
    @constraint(model, thermal_fixed_cost>=sum(thermal_units_N2[i].ConstTerm*is_on[i, t]+thermal_units_N2[i].StartUpCost*start_up[i, t]+thermal_units_N2[i].StartDownCost*start_down[i, t] for i in 1:N2 for t in 1:T))

    @variable(model, thermal_fuel_cost>=0)

    @variable(model, power[i in 1:N2, t in 0:T]>=0)

    @constraint(model,  [i in 1:N2, t in 0:T], power[i, t]>=thermal_units_N2[i].MinPower*is_on[i, t])
    @constraint(model,  [i in 1:N2, t in 0:T], power[i, t]<=thermal_units_N2[i].MaxPower*is_on[i, t])
    @constraint(model,  [i in 1:N2; thermal_units_N2[i].InitialPower!=nothing], power[i, 0]==thermal_units_N2[i].InitialPower)
    @constraint(model,  muup[i in 1:N2, t in 1:T], -(power[i, t]-power[i, t-1]) >= -((-thermal_units_N2[i].DeltaRampUp)*start_up[i, t]+(thermal_units_N2[i].MinPower+thermal_units_N2[i].DeltaRampUp)*is_on[i, t]-(thermal_units_N2[i].MinPower)*is_on[i, t-1]))
    @constraint(model,  mudown[i in 1:N2, t in 1:T], -(power[i, t-1]-power[i, t]) >= -((-thermal_units_N2[i].DeltaRampDown)*start_down[i, t]+(thermal_units_N2[i].MinPower+thermal_units_N2[i].DeltaRampDown)*is_on[i, t-1]-(thermal_units_N2[i].MinPower)*is_on[i, t]))
    @constraint(model, thermal_fuel_cost >= thermal_fixed_cost + sum(thermal_units_N2[i].LinearTerm*power[i, t] for i in 1:N2 for t in 1:T))

    @objective(model, Min, thermal_fuel_cost)
    
    power2 = Dict((i,t) => JuMP.variable_by_name(model, string("power[",i,",",t,"]")) for i in 1:N2 for t in 0:T)
    return relaxation2(model, power2)
end

function initialize_price_relaxation(instance)
    return priceRelaxationRO(decompose_problem_N1(instance), decompose_problem_N2(instance))
end