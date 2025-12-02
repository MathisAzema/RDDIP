mutable struct LagrangianRO
    model::Model
    theta::VariableRef
    upper_constraint::Union{Nothing, JuMP.ConstraintRef}
    uncertainty::Dict{Symbol, VariableRef}
    dual_var_states_1::Dict{Symbol, VariableRef}
    dual_var_states_2::Dict{Symbol, VariableRef}
    dual_var_uncertainty::Dict{Symbol, VariableRef}
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
    dual_var_states_1::Dict{Symbol, VariableRef}
    uncertainty::Dict{Symbol, VariableRef}
end

mutable struct subproblemRO
    model::Model
    states_1::Dict{Symbol, VariableRef}
    states_2::Dict{Symbol, VariableRef}
    uncertainty::Dict{Symbol, VariableRef}
    cstr_demand::Matrix{ConstraintRef}
end

mutable struct relaxation1
    model::Model
    muup::Matrix{ConstraintRef}
    mudown::Matrix{ConstraintRef}
    states::Dict{Symbol, VariableRef}
    power::Dict{Tuple{Int64, Int64}, VariableRef}
    power_shedding::Dict{Tuple{Int64, Int64}, VariableRef}
    power_curtailement::Dict{Tuple{Int64, Int64}, VariableRef}  
    flow::Dict{Tuple{Int64, Int64}, VariableRef}
end

mutable struct relaxation2
    model::Model
    power::Dict{Tuple{Int64, Int64}, VariableRef}
end

mutable struct priceProblemRO
    model::Model
    theta::VariableRef
    upper_constraint::Union{Nothing, JuMP.ConstraintRef}
    dual_var_states_1::Dict{Symbol, VariableRef}
    dual_var_states_2::Dict{Symbol, VariableRef}
    dual_demand::Matrix{VariableRef}
end

mutable struct priceRelaxationRO
    model1::relaxation1
    model2::relaxation2
    price_problem::priceProblemRO
end

mutable struct MasterPbRO
    model::Model
    states_1::Dict{Symbol, VariableRef}
end

mutable struct LagrangianStatesRO
    model::Model
    theta::VariableRef
    upper_constraint::Union{Nothing, JuMP.ConstraintRef}
    dual_var_states_1::Dict{Symbol, VariableRef}
    dual_var_states_2::Dict{Symbol, VariableRef}
    dual_var_uncertainty::Dict{Symbol, VariableRef}
end

struct optionsRO
    WorstCaseStrategy::String
end

Enumeration = optionsRO("Enumeration")
CuttingPlane = optionsRO("CuttingPlane")

mutable struct twoRO
    master_pb::MasterPbRO
    lagrangian::LagrangianRO
    subproblem::subproblemRO
    oracleContinuousRO::oracleContinuousRO
    priceRelaxation::priceRelaxationRO
    lagrangianStates::LagrangianStatesRO
    options::optionsRO
end

function master_RO_bin_problem_extended(instance, gap; silent=true)
    """
    Initial master problem in the risk neutral 3-bin formulation
    """

    model = Model(instance.optimizer)
    if silent
        set_silent(model)
    end

    set_optimizer_attribute(model, "Threads", 1)
    newgap=gap/100
    set_optimizer_attribute(model, "MIPGap", newgap)

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

    states_1 = Dict(Symbol("is_on[$i,$t]") => is_on[i,t] for i in 1:N1, t in 0:T)
    
    return MasterPbRO(model, states_1)
end

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

function builder_RO(instance; Γ=0)
    model = Model(instance.optimizer)

    set_silent(model)

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

    # @variable(model, uncertainty[t in 1:T]>=0)
    @variable(model, uncertainty[t in 1:T], Bin) #Binary is very more efficient here to learn \mu

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

    states_1 = Dict(Symbol("is_on[$i,$t]") => is_on[i,t] for i in 1:N1, t in 0:T)
    states_2 = Dict(Symbol("is_on[$i,$t]") => is_on[i,t] for i in N1+1:N, t in 0:T)
    uncertainty_var = Dict(Symbol("uncertainty[$t]") => uncertainty[t] for t in 1:T)

    cstr_demand = Matrix{ConstraintRef}(undef, T, length(Buses))
    for t in 1:T
        for b in Buses
            cstr_demand[t,b] = demand[t,b]
        end
    end

    return (model = model,
            states_1 = states_1,
            states_2 = states_2,
            uncertainty = uncertainty_var,
            demand=cstr_demand)
end

function Lagrangian_RO_problem(instance, gap; silent=true, Γ=0)
    """
    Initial master problem in the risk neutral 3-bin formulation
    """

    T= instance.TimeHorizon
    Next=instance.Next
    Buses=1:size(Next)[1]

    build = builder_RO(instance; Γ=Γ)
    model = build.model
    states_1 = build.states_1
    states_2 = build.states_2
    uncertainty_var = build.uncertainty

    undo_relax = JuMP.relax_integrality(model)
    new_model, _ = safe_copy(model)
    undo_relax()

    copy_states_1 = Dict()
    for (name, _) in states_1
        copy_states_1[name] = JuMP.variable_by_name(new_model, string(name))
    end
    copy_states_2 = Dict()
    for (name, _) in states_2
        copy_states_2[name] = JuMP.variable_by_name(new_model, string(name))
    end
    copy_uncertainty = Dict()
    for (name, _) in uncertainty_var
        copy_uncertainty[name] = JuMP.variable_by_name(new_model, string(name))
    end

    @constraint(new_model, fix_states_1[name in keys(copy_states_1)], copy_states_1[name] == 0.0)
    @constraint(new_model, fix_states_2[name in keys(copy_states_2)], copy_states_2[name] == 0.0)
    @constraint(new_model, fix_uncertainty[name in keys(copy_uncertainty)], copy_uncertainty[name] == 0.0)

    md=dualize(new_model; consider_constrained_variables=false, dual_names = DualNames("dual_var_", "dual_con_"))

    hexpr = JuMP.objective_function(md)
    theta = @variable(
        md,
        base_name = "theta",
    )
    upper_cstr = @constraint(md, theta <= hexpr)

    dual_var_states_1 = Dict(name => md[Symbol("dual_var_fix_states_1[$name]")] for name in keys(copy_states_1))
    dual_var_states_2 = Dict(name => md[Symbol("dual_var_fix_states_2[$name]")] for name in keys(copy_states_2))
    dual_var_uncertainty = Dict(name => md[Symbol("dual_var_fix_uncertainty[$name]")] for name in keys(copy_uncertainty))

    dual_demand = [md[Symbol("dual_var_demand[$t,$b]")] for t in 1:T, b in Buses]

    @variable(md, δ[t in 1:T])
    @variable(md, uncertainty[t in 1:T], Bin)
    ξ = Dict(Symbol("uncertainty[$t]") => uncertainty[t] for t in 1:T)
    @constraint(md, sum(uncertainty[t] for t in 1:T) <= Γ)
    max_D = [sum(d[t]*0.025*1.96 for d in instance.Demandbus) for t in 1:T]
    @constraint(md, [k in 1:T], δ[k] <= SHEDDING_COST * max_D[k] * uncertainty[k])
    @constraint(md, [k in 1:T], δ[k] >= -CURTAILEMENT_COST * max_D[k] * uncertainty[k])
    @constraint(md, [k in 1:T], δ[k] <= dual_var_uncertainty[Symbol("uncertainty[$k]")] + CURTAILEMENT_COST * max_D[k] * (1 - uncertainty[k]))
    @constraint(md, [k in 1:T], δ[k] >= dual_var_uncertainty[Symbol("uncertainty[$k]")] - SHEDDING_COST * max_D[k] * (1 - uncertainty[k]))
    JuMP.set_objective_function(
        md,
            @expression(md, theta + sum(δ)),
    )

    set_optimizer(md, instance.optimizer)    
    set_optimizer_attribute(md, "Threads", 1)
    set_optimizer_attribute(md, "TimeLimit", 20)
    newgap=gap/100
    set_optimizer_attribute(md, "MIPGap", newgap)
    if silent
        set_silent(md)
    end

    vars = [v for v in JuMP.all_variables(md) if JuMP.name(v) != "theta"]
    
    return LagrangianRO(md, theta, upper_cstr, ξ, dual_var_states_1, dual_var_states_2, dual_var_uncertainty, dual_demand, vars)
end

function subproblemRO(instance, gap, Γ)

    build = builder_RO(instance; Γ=Γ)
    model = build.model
    states_1 = build.states_1
    states_2 = build.states_2
    uncertainty_var = build.uncertainty
    demand = build.demand

    set_silent(model)

    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "TimeLimit", 10)
    newgap=gap/100
    set_optimizer_attribute(model, "MIPGap", newgap)

    return subproblemRO(model, states_1, states_2, uncertainty_var, demand)
end

function oracle_Continuous_RO_problem(instance; silent=true, Γ=0)
    """
    Initial master problem in the risk neutral 3-bin formulation
    """

    T= instance.TimeHorizon

    build = builder_RO(instance; Γ=Γ)
    model = build.model
    states_1 = build.states_1
    uncertainty_var = build.uncertainty

    undo_relax = JuMP.relax_integrality(model)
    new_model, _ = safe_copy(model)
    undo_relax()

    copy_states_1 = Dict()
    for (name, _) in states_1
        copy_states_1[name] = JuMP.variable_by_name(new_model, string(name))
    end

    copy_uncertainty = Dict()
    for (name, _) in uncertainty_var
        copy_uncertainty[name] = JuMP.variable_by_name(new_model, string(name))
    end

    @constraint(new_model, fix_states_1[name in keys(copy_states_1)], copy_states_1[name] == 0.0)
    @constraint(new_model, fix_uncertainty[name in keys(copy_uncertainty)], copy_uncertainty[name] == 0.0)

    md=dualize(new_model; consider_constrained_variables=false, dual_names = DualNames("dual_var_", "dual_con_"))

    hexpr = JuMP.objective_function(md)
    theta = @variable(
        md,
        base_name = "theta",
    )
    @constraint(md, theta <= hexpr)

    dual_var_states_1 = Dict(name => md[Symbol("dual_var_fix_states_1[$name]")] for name in keys(copy_states_1))
    dual_var_uncertainty = Dict(name => md[Symbol("dual_var_fix_uncertainty[$name]")] for name in keys(copy_uncertainty))
    
    @variable(md, δ[t in 1:T])
    @variable(md, uncertainty[t in 1:T], Bin)
    ξ = Dict(Symbol("uncertainty[$t]") => uncertainty[t] for t in 1:T)
    @constraint(md, sum(uncertainty[t] for t in 1:T) <= Γ)
    max_D = [sum(d[t]*0.025*1.96 for d in instance.Demandbus) for t in 1:T]
    @constraint(md, [k in 1:T], δ[k] <= SHEDDING_COST * max_D[k] * uncertainty[k])
    @constraint(md, [k in 1:T], δ[k] >= -CURTAILEMENT_COST * max_D[k] * uncertainty[k])
    @constraint(md, [k in 1:T], δ[k] <= dual_var_uncertainty[Symbol("uncertainty[$k]")] + CURTAILEMENT_COST * max_D[k] * (1 - uncertainty[k]))
    @constraint(md, [k in 1:T], δ[k] >= dual_var_uncertainty[Symbol("uncertainty[$k]")] - SHEDDING_COST * max_D[k] * (1 - uncertainty[k]))
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
    
    return oracleContinuousRO(md, dual_var_states_1, ξ)
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

function decompose_problem_N1(instance)
    """
    Initial master problem in the risk neutral 3-bin formulation
    """

    model = Model(instance.optimizer)
    set_silent(model)

    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "TimeLimit", 5)

    T= instance.TimeHorizon    
    N1 = instance.N1
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    Numlines=length(instance.Lines)
    thermal_units=instance.Thermalunits
    thermal_units_N1 = [unit for unit in thermal_units if unit.name <= N1]

    @variable(model, is_on[i in 1:N1, t in 0:T]>=0)
    @variable(model, start_up[i in 1:N1, t in 1:T]>=0)
    @variable(model, start_down[i in 1:N1, t in 1:T]>=0)

    @constraint(model, [i in 1:N1, t in 0:T], is_on[i,t]<=1)
    @constraint(model, [i in 1:N1, t in 1:T], start_up[i,t]<=1)
    @constraint(model, [i in 1:N1, t in 1:T], start_down[i,t]<=1)

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

    states_1 = Dict(Symbol("is_on[$i,$t]") => is_on[i,t] for i in 1:N1, t in 0:T)
    power1 = Dict((i,t) => JuMP.variable_by_name(model, string("power[",i,",",t,"]")) for i in 1:N1 for t in 0:T)
    power_shedding1 = Dict((b,t) => JuMP.variable_by_name(model, string("power_shedding[",b,",",t,"]")) for b in Buses for t in 1:T)
    power_curtailement1 = Dict((b,t) => JuMP.variable_by_name(model, string("power_curtailement[",b,",",t,"]")) for b in Buses for t in 1:T)

    flow1 = Dict((l,t) => JuMP.variable_by_name(model, string("flow[",l,",",t,"]")) for l in 1:Numlines for t in 1:T)
    
    return relaxation1(model, cstr_muup, cstr_mudown, states_1, power1, power_shedding1, power_curtailement1, flow1)
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
    return priceRelaxationRO(decompose_problem_N1(instance), decompose_problem_N2(instance), get_price_problem(instance))
end

function get_price_problem(instance)
    """
    Initial master problem in the risk neutral 3-bin formulation
    """

    model = Model(instance.optimizer)

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

    @constraint(model,  demand[t in 1:T, b in Buses], sum(power[unit.name, t] for unit in thermal_units if unit.Bus==b)+power_shedding[b,t]-power_curtailement[b,t] + sum(flow[line.id, t] for line in Lines if line.b2==b) - sum(flow[line.id, t] for line in Lines if line.b1==b) == 0.0)

    @objective(model, Min, thermal_fuel_cost)

    states_1 = Dict(Symbol("is_on[$i,$t]") => is_on[i,t] for i in 1:N1, t in 0:T)
    states_2 = Dict(Symbol("is_on[$i,$t]") => is_on[i,t] for i in N1+1:N, t in 0:T)

    undo_relax = JuMP.relax_integrality(model)
    new_model, _ = safe_copy(model)
    undo_relax()

    copy_states_1 = Dict()
    for (name, _) in states_1
        copy_states_1[name] = JuMP.variable_by_name(new_model, string(name))
    end
    copy_states_2 = Dict()
    for (name, _) in states_2
        copy_states_2[name] = JuMP.variable_by_name(new_model, string(name))
    end

    @constraint(new_model, fix_states_1[name in keys(copy_states_1)], copy_states_1[name] == 0.0)
    @constraint(new_model, fix_states_2[name in keys(copy_states_2)], copy_states_2[name] == 0.0)

    md=dualize(new_model; consider_constrained_variables=false, dual_names = DualNames("dual_var_", "dual_con_"))

    hexpr = JuMP.objective_function(md)
    theta = @variable(
        md,
        base_name = "theta",
    )
    upper_cstr = @constraint(md, theta <= hexpr)

    dual_var_states_1 = Dict(name => md[Symbol("dual_var_fix_states_1[$name]")] for name in keys(copy_states_1))
    dual_var_states_2 = Dict(name => md[Symbol("dual_var_fix_states_2[$name]")] for name in keys(copy_states_2))

    dual_demand = [md[Symbol("dual_var_demand[$t,$b]")] for t in 1:T, b in Buses]

    JuMP.set_objective_function(
        md,
            @expression(md, theta),
    )

    set_optimizer(md, instance.optimizer)    
    set_optimizer_attribute(md, "Threads", 1)
    set_optimizer_attribute(md, "TimeLimit", 10)

    set_silent(md)
    
    return priceProblemRO(md, theta, upper_cstr, dual_var_states_1, dual_var_states_2, dual_demand)
end


function LagrangianStatesRO_problem(instance; silent=true, Γ=0)
    """
    Initial master problem in the risk neutral 3-bin formulation
    """

    T= instance.TimeHorizon
    Next=instance.Next
    Buses=1:size(Next)[1]

    build = builder_RO(instance; Γ=Γ)
    model = build.model
    states_1 = build.states_1
    states_2 = build.states_2
    uncertainty_var = build.uncertainty

    undo_relax = JuMP.relax_integrality(model)
    new_model, _ = safe_copy(model)
    undo_relax()

    copy_states_1 = Dict()
    for (name, _) in states_1
        copy_states_1[name] = JuMP.variable_by_name(new_model, string(name))
    end
    copy_states_2 = Dict()
    for (name, _) in states_2
        copy_states_2[name] = JuMP.variable_by_name(new_model, string(name))
    end
    copy_uncertainty = Dict()
    for (name, _) in uncertainty_var
        copy_uncertainty[name] = JuMP.variable_by_name(new_model, string(name))
    end

    @constraint(new_model, fix_states_1[name in keys(copy_states_1)], copy_states_1[name] == 0.0)
    @constraint(new_model, fix_states_2[name in keys(copy_states_2)], copy_states_2[name] == 0.0)
    @constraint(new_model, fix_uncertainty[name in keys(copy_uncertainty)], copy_uncertainty[name] == 0.0)

    md=dualize(new_model; consider_constrained_variables=false, dual_names = DualNames("dual_var_", "dual_con_"))

    hexpr = JuMP.objective_function(md)
    theta = @variable(
        md,
        base_name = "theta",
    )
    upper_cstr = @constraint(md, theta <= hexpr)

    dual_var_states_1 = Dict(name => md[Symbol("dual_var_fix_states_1[$name]")] for name in keys(copy_states_1))
    dual_var_states_2 = Dict(name => md[Symbol("dual_var_fix_states_2[$name]")] for name in keys(copy_states_2))
    dual_var_uncertainty = Dict(name => md[Symbol("dual_var_fix_uncertainty[$name]")] for name in keys(copy_uncertainty))

    JuMP.set_objective_function(
        md,
            @expression(md, theta),
    )

    set_optimizer(md, instance.optimizer)    
    set_optimizer_attribute(md, "Threads", 1)
    set_optimizer_attribute(md, "TimeLimit", 10)

    set_silent(md)
    
    return LagrangianStatesRO(md, theta, upper_cstr, dual_var_states_1, dual_var_states_2, dual_var_uncertainty)
end