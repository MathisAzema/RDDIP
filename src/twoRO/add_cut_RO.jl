mutable struct ResultsPriceRelaxation
    obj::Float64
    dual_muup::Matrix{Float64}
    dual_mudown::Matrix{Float64}
    price_demand::Matrix{Float64}
    gradient::Matrix{Float64}
end

struct cutRO
    intercept::Float64
    is_on::Dict{Tuple{Int64, Int64}, Float64}
    start_up::Dict{Tuple{Int64, Int64}, Float64}
    start_down::Dict{Tuple{Int64, Int64}, Float64}
    uncertainty::Vector{Float64}
    dual_var_is_on::Dict{Tuple{Int64, Int64}, Float64}
    dual_var_start_up::Dict{Tuple{Int64, Int64}, Float64}
    dual_var_start_down::Dict{Tuple{Int64, Int64}, Float64}
    dual_var_uncertainty::Vector{Float64}
end


function add_cut_RO(cb_data, master_pb::JuMP.Model, oracle_pb::oracleRO, instance::Instance, Time_subproblem, solution_x::Vector{Matrix{Float64}}, solution_gamma::Dict{Tuple{Int64, Int64, Int64}, Float64}; gap)
    """
    Add Benders' cut
    """
    T= instance.TimeHorizon
    thermal_units=instance.Thermalunits
    N = instance.N

    dual_var_is_on = oracle_pb.dual_var_is_on
    dual_var_start_up = oracle_pb.dual_var_start_up
    dual_var_start_down = oracle_pb.dual_var_start_down

    thermal_cost=master_pb[:thermal_cost]
    thermal_fixed_cost=master_pb[:thermal_fixed_cost]

    thermal_cost_val=callback_value(cb_data, thermal_cost)
    thermal_fixed_cost_val=callback_value(cb_data, thermal_fixed_cost)

    solution_is_on = solution_x[1]
    solution_start_up = solution_x[2]
    solution_start_down = solution_x[3]

    primal_obj = JuMP.objective_function(oracle_pb.model)

    JuMP.set_objective_function(
        oracle_pb.model,
        @expression(oracle_pb.model, primal_obj + sum(
            dual_var_is_on[i,t] * solution_is_on[i,t+1] for
            i in 1:N for t in 0:T)) + sum(
            dual_var_start_up[i,t] * solution_start_up[i,t] for
            i in 1:N for t in 1:T) + sum(
            dual_var_start_down[i,t] * solution_start_down[i,t] for
            i in 1:N for t in 1:T)
    )

    start =time()

    JuMP.optimize!(oracle_pb.model)

    computation_time = time() - start
    println(computation_time)

    push!(Time_subproblem, computation_time)

    second_stage_cost = JuMP.objective_value(oracle_pb.model)

    current_intervals=Dict{Int, Vector{Tuple{Int, Int}}}()
    for unit in thermal_units
        current_intervals[unit.name] = Vector{Tuple{Int, Int}}[]
        for (a,b) in unit.intervals
            if solution_gamma[unit.name, a,b]>=0.9 
                push!(current_intervals[unit.name], (a,b))
            end
        end
    end

    if thermal_fixed_cost_val+second_stage_cost >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
        _add_optimality_cuts_RO(cb_data, master_pb, oracle_pb, instance, solution_gamma, current_intervals)
    end

    JuMP.set_objective_function(oracle_pb.model, primal_obj)

    return thermal_fixed_cost_val+second_stage_cost, second_stage_cost
end


function _add_optimality_cuts_RO(cb_data, master_pb, oracle_pb::oracleRO, instance::Instance, solution_gamma::Dict{Tuple{Int64, Int64, Int64}, Float64}, current_intervals::Dict{Int, Vector{Tuple{Int, Int}}})
    T= instance.TimeHorizon
    gamma=master_pb[:gamma]
    thermal_cost=master_pb[:thermal_cost]
    gamma_val=solution_gamma
    thermal_units=instance.Thermalunits
    Costab=Dict{Tuple{Int, Int, Int}, Float64}()
    Vab=zeros(T)
    Wab=zeros(T)
    price_unit=zeros(T)

    muup = JuMP.value.(oracle_pb.dual_up)
    mudown = JuMP.value.(oracle_pb.dual_down)
    price = JuMP.value.(oracle_pb.dual_demand)

    for unit in thermal_units
        price_unit=price[:, unit.Bus]
        for (a,b) in unit.intervals
            Costab[unit.name,a,b] = heuristic_cost(unit, a, b, price_unit, gamma_val, muup, mudown, current_intervals[unit.name], true, Vab, Wab)
        end
    end

    intercept = JuMP.objective_value(oracle_pb.model) - sum(Costab[unit.name,a,b] for unit in thermal_units for (a,b) in current_intervals[unit.name])

    cstr=@build_constraint(sum(gamma[unit.name,[a,b]]*Costab[unit.name,a,b] for unit in thermal_units for (a,b) in unit.intervals)+intercept<=thermal_cost)
    MOI.submit(master_pb, MOI.LazyConstraint(cb_data), cstr)

end

function get_worst_case_RO_lagrangian(twoROmodel::twoRO, solution_xN1::Vector{Matrix{Float64}}, solution_xN2::Vector{Matrix{Float64}}, instance::Instance; cut::Union{cutRO, Nothing} = nothing)
    """
    Get worst-case cost from Lagrangian relaxation
    """
    lagrangian = twoROmodel.lagrangian

    dual_var_is_on = lagrangian.dual_var_is_on
    dual_var_start_up = lagrangian.dual_var_start_up
    dual_var_start_down = lagrangian.dual_var_start_down

    T= instance.TimeHorizon
    N1 = instance.N1
    N2 = instance.N - N1

    solution_is_on = solution_xN1[1]
    solution_start_up = solution_xN1[2]
    solution_start_down = solution_xN1[3]

    if cut!=nothing
        for i in 1:N1
            for t in 0:T
                JuMP.fix(dual_var_is_on[i,t], cut.dual_var_is_on[i,t]; force=true)
            end
        end
    end

    primal_obj = JuMP.objective_function(lagrangian.model)

    JuMP.set_objective_function(
        lagrangian.model,
        @expression(lagrangian.model, primal_obj + sum(
            dual_var_is_on[i,t] * solution_is_on[i,t+1] for
            i in 1:N1 for t in 0:T)) + sum(
            dual_var_start_up[i,t] * solution_start_up[i,t] for
            i in 1:N1 for t in 1:T) + sum(
            dual_var_start_down[i,t] * solution_start_down[i,t] for
            i in 1:N1 for t in 1:T)
    )

    solution_is_on_N2 = solution_xN2[1]
    solution_start_up_N2 = solution_xN2[2]
    solution_start_down_N2 = solution_xN2[3]

    cstr = lagrangian.upper_constraint
    for i in 1:N2
        for t in 0:T
            value = solution_is_on_N2[i,t+1]
            var = lagrangian.dual_var_is_onN2[i,t]
            set_normalized_coefficient(cstr, var, -value)
        end
    end

    for i in 1:N2
        for t in 1:T
            value = solution_start_up_N2[i,t]
            var = lagrangian.dual_var_start_upN2[i,t]
            set_normalized_coefficient(cstr, var, -value)
            value = solution_start_down_N2[i,t]
            var = lagrangian.dual_var_start_downN2[i,t]
            set_normalized_coefficient(cstr, var, -value)
        end
    end

    JuMP.optimize!(lagrangian.model)

    worst_case_cost_bound = JuMP.objective_bound(lagrangian.model)

    worst_case = Dict(name => JuMP.value(var) for (name, var) in lagrangian.uncertainty)

    sol_dual_var_is_on = Dict((i,t) => JuMP.value(dual_var_is_on[i,t]) for i in 1:N1 for t in 0:T)
    sol_dual_var_start_up = Dict((i,t) => JuMP.value(dual_var_start_up[i,t]) for i in 1:N1 for t in 1:T)
    sol_dual_var_start_down = Dict((i,t) => JuMP.value(dual_var_start_down[i,t]) for i in 1:N1 for t in 1:T)

    sol_dual_var = [sol_dual_var_is_on, sol_dual_var_start_up, sol_dual_var_start_down]

    sol_dual_uncertainty = JuMP.value.(lagrangian.dual_var_uncertainty)

    theta_val = JuMP.value(lagrangian.theta)

    dual_demand = JuMP.value.(lagrangian.dual_demand)

    if cut!=nothing
        for i in 1:N1
            for t in 0:T
                JuMP.unfix(dual_var_is_on[i,t])
            end
        end
    end

    JuMP.set_objective_function(lagrangian.model, primal_obj)

    return worst_case_cost_bound, worst_case, theta_val, sol_dual_var, sol_dual_uncertainty, dual_demand
end

function get_worst_case_RO_continuous(twoROmodel::twoRO, states_1::Dict{Symbol, Float64})
    """
    Get worst-case cost from Lagrangian relaxation
    """
    oraclecontinuous = twoROmodel.oracleContinuousRO

    dual_var_states_1 = oraclecontinuous.dual_var_states_1

    primal_obj = JuMP.objective_function(oraclecontinuous.model)

    JuMP.set_objective_function(
        oraclecontinuous.model,
        @expression(oraclecontinuous.model, primal_obj + sum(
            dual_var_states_1[name] * value for
            (name,value) in states_1))
    )

    JuMP.optimize!(oraclecontinuous.model)

    worst_case = Dict(name => round(JuMP.value(var)) for (name, var) in oraclecontinuous.uncertainty)

    JuMP.set_objective_function(oraclecontinuous.model, primal_obj)

    return worst_case
end

function solve_second_stage_RO(twoROmodel::twoRO, states_1::Dict{Symbol, Float64}, uncertainty::Dict{Symbol, Float64})
    """
    Get worst-case cost from Lagrangian relaxation
    """
    subproblem = twoROmodel.subproblem

    for (name, var) in subproblem.states_1
        fix(var, states_1[name]; force=true)
    end

    for (name, var) in subproblem.uncertainty
        fix(var, uncertainty[name]; force=true)
    end

    JuMP.optimize!(subproblem.model)


    obj = JuMP.objective_value(subproblem.model)
    bound = JuMP.objective_bound(subproblem.model)

    states_2_val = Dict{Symbol, Float64}(name => round(JuMP.value(var)) for (name, var) in subproblem.states_2)

    for (_, var) in subproblem.states_1
        JuMP.unfix(var)
    end

    for (_, var) in subproblem.uncertainty
        JuMP.unfix(var)
    end

    return (objective = obj, bound = bound, states_2_val = states_2_val)
end

function solve_second_stage_RO_lagrangian(twoROmodel::twoRO, instance::Instance, dual_state::Vector{Dict{Tuple{Int64, Int64}, Float64}}, dual_uncertainty::Vector{Float64})
    """
    Get worst-case cost from Lagrangian relaxation
    """
    subproblem = twoROmodel.subproblem

    is_on = subproblem.is_on
    start_up = subproblem.start_up
    start_down = subproblem.start_down
    uncertainty = subproblem.uncertainty

    T= instance.TimeHorizon
    N = instance.N
    N1 = instance.N1
    N2 = instance.N - N1

    primal_obj = JuMP.objective_function(subproblem.model)

    JuMP.set_objective_function(
        subproblem.model,
        @expression(subproblem.model, primal_obj - sum(
            dual_state[1][i,t] * is_on[i,t] for
            i in 1:N1 for t in 0:T) - sum(
            dual_state[2][i,t] * start_up[i,t] for
            i in 1:N1 for t in 1:T) - sum(
            dual_state[3][i,t] * start_down[i,t] for
            i in 1:N1 for t in 1:T) - sum(
            dual_uncertainty[t] * uncertainty[t] for t in 1:T))
    )

    JuMP.optimize!(subproblem.model)

    obj = JuMP.objective_value(subproblem.model)
    bound = JuMP.objective_bound(subproblem.model)

    worst_case = [round(JuMP.value(uncertainty[t])) for t in 1:T]

    sol_is_on = Dict((i,t) => round(JuMP.value(is_on[i,t])) for i in 1:N1 for t in 0:T)
    sol_start_up = Dict((i,t) => round(JuMP.value(start_up[i,t])) for i in 1:N1 for t in 1:T)
    sol_start_down = Dict((i,t) => round(JuMP.value(start_down[i,t])) for i in 1:N1 for t in 1:T)

    sol_state = [sol_is_on, sol_start_up, sol_start_down]

    JuMP.set_objective_function(subproblem.model, primal_obj)
    return obj, bound, sol_state, worst_case
end

function solve_second_stage_RO_lagrangianUncertainty(twoROmodel::twoRO, states_1::Dict{Symbol, Float64}, dual_uncertainty::Dict{Symbol, Float64})
    """
    Get worst-case cost from Lagrangian relaxation
    """
    subproblem = twoROmodel.subproblem

    uncertainty = subproblem.uncertainty

    for (name, var) in subproblem.states_1
        fix(var, states_1[name]; force=true)
    end

    primal_obj = JuMP.objective_function(subproblem.model)

    JuMP.set_objective_function(
        subproblem.model,
        @expression(subproblem.model, primal_obj - sum(
            dual_uncertainty[name] * uncertainty[name] for (name, _) in dual_uncertainty))
    )

    start =time()
    JuMP.optimize!(subproblem.model)
    # println(time() - start)

    obj = JuMP.objective_value(subproblem.model)

    worst_case = Dict(name => round(JuMP.value(var)) for (name, var) in uncertainty)

    JuMP.set_objective_function(subproblem.model, primal_obj)
    
    for (_, var) in subproblem.states_1
        JuMP.unfix(var)
    end

    return (objective = obj, worst_case = worst_case)
end

function solve_second_stage_RO_lagrangianStates(twoROmodel::twoRO, states_1_val::Dict{Symbol, Float64}, dual_states_1::Dict{Symbol, Float64}, solution_uncertainty::Dict{Symbol, Float64})
    """
    Get worst-case cost from Lagrangian relaxation
    """
    subproblem = twoROmodel.subproblem

    states_1 = subproblem.states_1

    for (name, var) in subproblem.uncertainty
        fix(var, solution_uncertainty[name]; force=true)
    end

    primal_obj = JuMP.objective_function(subproblem.model)

    JuMP.set_objective_function(
        subproblem.model,
        @expression(subproblem.model, primal_obj - sum(
            dual_states_1[name] * states_1[name] for (name, _) in dual_states_1))
    )

    start = time()
    JuMP.optimize!(subproblem.model)
    # println(time() - start)

    obj = JuMP.objective_value(subproblem.model)
    bound = JuMP.objective_bound(subproblem.model)

    states_1_case = Dict(name => round(JuMP.value(var)) for (name, var) in states_1)

    JuMP.set_objective_function(subproblem.model, primal_obj)
    
    for (_, var) in subproblem.uncertainty
        JuMP.unfix(var)
    end

    return (objective = obj, bound = bound, states_1 = states_1_case)
end

function add_cut_RO_bin(cb_data, twoROmodel::twoRO, instance::Instance, Time_subproblem, states_1::Dict{Symbol, Float64}, solution_gamma::Dict{Tuple{Int64, Int64, Int64}, Float64}, Γ; gap)
    """
    Add Benders' cut
    """
    # println("Add cut RO bin")
    thermal_units=instance.Thermalunits
    N1 = instance.N1
    thermal_units_N1=thermal_units[1:N1]    

    master_pb = twoROmodel.master_pb.model

    N1 = instance.N1

    thermal_cost=master_pb[:thermal_cost]
    thermal_fixed_cost=master_pb[:thermal_fixed_cost]

    thermal_cost_val=callback_value(cb_data, thermal_cost)
    thermal_fixed_cost_val=callback_value(cb_data, thermal_fixed_cost)

    current_intervals=Dict{Int, Vector{Tuple{Int, Int}}}()
    for unit in thermal_units_N1
        current_intervals[unit.name] = Vector{Tuple{Int, Int}}[]
        for (a,b) in unit.intervals
            if solution_gamma[unit.name, a,b]>=0.9 
                push!(current_intervals[unit.name], (a,b))
            end
        end
    end


    worst_case_continuous = get_worst_case_RO_continuous(twoROmodel, states_1)

    result_continuous = solve_second_stage_RO(twoROmodel, states_1, worst_case_continuous)

    add_cut = false

    update_UB = false

    if thermal_fixed_cost_val+result_continuous.objective >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
        
        # price_demand = get_price_demand_RO(twoROmodel, states_1, worst_case_continuous, instance)

        # results_price_relaxation = solve_price_relaxation(twoROmodel, instance, price_demand, states_1, worst_case_continuous)

        results_price_relaxation = get_optimal_price_demand(twoROmodel, states_1, result_continuous.states_2_val, worst_case_continuous, instance)

        worst_case_cost_obj = results_price_relaxation.obj

        if thermal_fixed_cost_val+results_price_relaxation.obj >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
            _add_optimality_cuts_RO_bin(cb_data, master_pb, results_price_relaxation, instance, solution_gamma, current_intervals)
            add_cut = true
        end

        println(("heur", add_cut, thermal_fixed_cost_val, thermal_fixed_cost_val + results_price_relaxation.obj, thermal_fixed_cost_val + result_continuous.objective, thermal_fixed_cost_val+thermal_cost_val))

    end

    if !add_cut
        update_UB = true
        # println("Compute worst-case Lagrangian Uncertainty ", thermal_fixed_cost_val + thermal_cost_val, " ", thermal_fixed_cost_val+obj_continuous)

        start =time()
        if twoROmodel.options == CuttingPlane
            results_lagrangian = get_worst_case_RO_lagrangianUncertainty_callback(twoROmodel, states_1, result_continuous.states_2_val)
        elseif twoROmodel.options == Enumeration
            results_lagrangian = get_worst_case_RO_enumeration(twoROmodel, states_1, instance, Γ)
        end
        push!(Time_subproblem, time() - start)

        # worst_case_cost_obj = results_lagrangian.objective

        # price_demand = get_price_demand_RO(twoROmodel, states_1, results_lagrangian.worst_case, instance)

        # results_price_relaxation = solve_price_relaxation(twoROmodel, instance, price_demand, states_1, results_lagrangian.worst_case)

        results_price_relaxation = get_optimal_price_demand(twoROmodel, states_1, result_continuous.states_2_val, results_lagrangian.worst_case, instance)

        if thermal_fixed_cost_val+results_price_relaxation.obj >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
            _add_optimality_cuts_RO_bin(cb_data, master_pb, results_price_relaxation, instance, solution_gamma, current_intervals)
            add_cut = true
        end

        worst_case_cost_obj = max(results_price_relaxation.obj, thermal_cost_val)
        println(("callback", add_cut, thermal_fixed_cost_val, thermal_fixed_cost_val + results_price_relaxation.obj, thermal_fixed_cost_val + results_lagrangian.objective, thermal_fixed_cost_val+thermal_cost_val))
        println(thermal_fixed_cost_val+worst_case_cost_obj)
        # println(("callback2", add_cut, thermal_fixed_cost_val, results_price_relaxation.obj, results_lagrangian.objective, thermal_cost_val))

        #Calcul des coefficients de la coupe

        # results_SB = get_SB_cut_RO(twoROmodel, states_1, results_lagrangian.worst_case)

        # # println((thermal_fixed_cost_val, worst_case_cost_obj, results_SB.objective, thermal_cost_val))
        # if thermal_fixed_cost_val+results_SB.objective >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
        #     _add_SB_cut(cb_data, twoROmodel, states_1, results_SB.objective, results_SB.sol_dual_var_states_1)
        #     add_cut = true
        # end

        # worst_case_cost_obj = max(results_price_relaxation.obj, results_SB.objective)

        # results_L = get_lagrangian_cut(twoROmodel, states_1, result_continuous.states_2_val, results_lagrangian.worst_case)

        # # println((thermal_fixed_cost_val, worst_case_cost_obj, results_L.objective, thermal_cost_val))
        # if thermal_fixed_cost_val+results_L.objective >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
        #     _add_SB_cut(cb_data, twoROmodel, states_1, results_L.objective, results_L.sol_dual_var_states_1)
        #     add_cut = true
        # end
    end

    return update_UB, thermal_fixed_cost_val+worst_case_cost_obj, worst_case_cost_obj
    
end

function get_worst_case_RO_enumeration(twoROmodel::twoRO, states_1::Dict{Symbol, Float64}, instance::Instance, Γ::Int64)
    T = instance.TimeHorizon
    uncertainty_set = generate_all_Γ_tuple(T, Γ)
    worst_case_cost = -Inf
    worst_case = Dict{Int, Float64}()
    for uncertainty in uncertainty_set
        uncertainty_dict = Dict(Symbol("uncertainty[$t]") => uncertainty[t]*1.0 for t in 1:T)
        results = solve_second_stage_RO(twoROmodel, states_1, uncertainty_dict)
        if results.objective > worst_case_cost
            worst_case_cost = results.objective
            worst_case = uncertainty_dict
        end
    end
    return (objective = worst_case_cost, worst_case = worst_case)
end

function _add_lagrangian_optimality_cuts_RO_bin(cb_data, master_pb, instance::Instance, solution_xN1::Vector{Matrix{Float64}}, worst_case_cost_obj::Float64, sol_dual_var::Vector{Dict{Tuple{Int64, Int64}, Float64}})

    T= instance.TimeHorizon
    N1 = instance.N1
    is_on = master_pb[:is_on]
    start_up = master_pb[:start_up]
    start_down = master_pb[:start_down]
    thermal_cost=master_pb[:thermal_cost]

    cstr=@build_constraint(worst_case_cost_obj + sum(sol_dual_var[1][i,t] * (is_on[i,t] - solution_xN1[1][i,t+1]) + sol_dual_var[2][i,t] * (start_up[i,t] - solution_xN1[2][i,t]) + sol_dual_var[3][i,t] * (start_down[i,t] - solution_xN1[3][i,t])  for i in 1:N1 for t in 1:T) <= thermal_cost) #attention 0 ou 1:T ?
    MOI.submit(master_pb, MOI.LazyConstraint(cb_data), cstr)
    # println("add_lagrangian_cut :", worst_case_cost_obj)
    # println(worst_case_cost_obj + sum(sol_dual_var[1][i,t] * (1 - solution_xN1[1][i,t+1]) for i in 1:N1 for t in 1:T))
end

function _add_SB_cut(cb_data, twoROmodel::twoRO, states_1_val::Dict{Symbol, Float64}, cost_SB::Float64, sol_dual_states_1::Dict{Symbol, Float64})

    states_1 = twoROmodel.master_pb.states_1
    thermal_cost=twoROmodel.master_pb.model[:thermal_cost]

    cstr=@build_constraint(cost_SB + sum(sol_dual_states_1[name] * (var - states_1_val[name]) for (name, var) in states_1) <= thermal_cost) #attention 0 ou 1:T ?
    MOI.submit(twoROmodel.master_pb.model, MOI.LazyConstraint(cb_data), cstr)
end


function _add_optimality_cuts_RO_bin(cb_data, master_pb, results_price_relaxation::ResultsPriceRelaxation, instance::Instance, solution_gamma::Dict{Tuple{Int64, Int64, Int64}, Float64}, current_intervals::Dict{Int, Vector{Tuple{Int, Int}}})
    T= instance.TimeHorizon
    gamma=master_pb[:gamma]
    thermal_cost=master_pb[:thermal_cost]
    gamma_val=solution_gamma
    N1 = instance.N1
    thermal_units=instance.Thermalunits
    thermal_units_N1 = [thermal_units[i] for i in 1:N1]
    Costab=Dict{Tuple{Int, Int, Int}, Float64}()
    Vab=zeros(T)
    Wab=zeros(T)
    price_unit=zeros(T)

    muup = results_price_relaxation.dual_muup
    mudown = results_price_relaxation.dual_mudown
    price = results_price_relaxation.price_demand
    obj_relaxation = results_price_relaxation.obj

    for unit in thermal_units_N1
        price_unit=price[:, unit.Bus]
        for (a,b) in unit.intervals
            Costab[unit.name,a,b] = heuristic_cost(unit, a, b, price_unit, gamma_val, muup, mudown, current_intervals[unit.name], true, Vab, Wab)
        end
    end

    intercept = obj_relaxation - sum(Costab[unit.name,a,b] for unit in thermal_units_N1 for (a,b) in current_intervals[unit.name])

    cstr=@build_constraint(sum(gamma[unit.name,[a,b]]*Costab[unit.name,a,b] for unit in thermal_units_N1 for (a,b) in unit.intervals)+intercept<=thermal_cost)
    MOI.submit(master_pb, MOI.LazyConstraint(cb_data), cstr)

    # println("price_relaxation_cut: ",sum(gamma_val[unit.name,a,b]*Costab[unit.name,a,b] for unit in thermal_units_N1 for (a,b) in unit.intervals)+intercept)

end


function get_worst_case_RO_lagrangian_callback(twoROmodel::twoRO, solution_xN1::Vector{Matrix{Float64}}, solution_xN2::Vector{Matrix{Float64}}, instance::Instance)
    """
    Get worst-case cost from Lagrangian relaxation
    """
    lagrangian = twoROmodel.lagrangian

    dual_var_is_on = lagrangian.dual_var_is_on
    dual_var_start_up = lagrangian.dual_var_start_up
    dual_var_start_down = lagrangian.dual_var_start_down

    dual_var_uncertainty = lagrangian.dual_var_uncertainty

    T= instance.TimeHorizon
    N1 = instance.N1
    N2 = instance.N - N1

    solution_is_on = solution_xN1[1]
    solution_start_up = solution_xN1[2]
    solution_start_down = solution_xN1[3]


    primal_obj = JuMP.objective_function(lagrangian.model)

    new_obj_function = @expression(lagrangian.model, primal_obj + sum(
            dual_var_is_on[i,t] * solution_is_on[i,t+1] for
            i in 1:N1 for t in 0:T)) + sum(
            dual_var_start_up[i,t] * solution_start_up[i,t] for
            i in 1:N1 for t in 1:T) + sum(
            dual_var_start_down[i,t] * solution_start_down[i,t] for
            i in 1:N1 for t in 1:T)

    JuMP.set_objective_function(
        lagrangian.model,
        new_obj_function
    )

    solution_is_on_N2 = solution_xN2[1]
    solution_start_up_N2 = solution_xN2[2]
    solution_start_down_N2 = solution_xN2[3]

    cstr = lagrangian.upper_constraint
    for i in 1:N2
        for t in 0:T
            value = solution_is_on_N2[i,t+1]
            var = lagrangian.dual_var_is_onN2[i,t]
            set_normalized_coefficient(cstr, var, -value)
        end
    end

    for i in 1:N2
        for t in 1:T
            value = solution_start_up_N2[i,t]
            var = lagrangian.dual_var_start_upN2[i,t]
            set_normalized_coefficient(cstr, var, -value)
            value = solution_start_down_N2[i,t]
            var = lagrangian.dual_var_start_downN2[i,t]
            set_normalized_coefficient(cstr, var, -value)
        end
    end

    list_cuts = cutRO[]
    LB=0.0
    k = 0

    function my_callback_function(cb_data, cb_where::Cint)

        if cb_where==GRB_CB_MIPSOL
            k+=1
    
            Gurobi.load_callback_variable_primal(cb_data, cb_where)

            sol_dual_uncertainty = callback_value.(cb_data, dual_var_uncertainty)

            sol_dual_var_is_on = Dict((i,t) => callback_value(cb_data, dual_var_is_on[i,t]) for i in 1:N1 for t in 0:T)
            sol_dual_var_start_up = Dict((i,t) => callback_value(cb_data, dual_var_start_up[i,t]) for i in 1:N1 for t in 1:T)
            sol_dual_var_start_down = Dict((i,t) => callback_value(cb_data, dual_var_start_down[i,t]) for i in 1:N1 for t in 1:T)

            sol_dual_state = [sol_dual_var_is_on, sol_dual_var_start_up, sol_dual_var_start_down]

            theta_val = callback_value(cb_data, lagrangian.theta)

            vars_val = callback_value.(cb_data, lagrangian.vars)

            current_obj = callback_value(cb_data, new_obj_function)

            obj, bound, sol_state, worst_case = solve_second_stage_RO_lagrangian(twoROmodel, instance, sol_dual_state, sol_dual_uncertainty)

            if obj<= 0.999999*theta_val

                intercept = obj + sum(
                    sol_state[1][i,t] * sol_dual_var_is_on[i,t] for
                    i in 1:N1 for t in 0:T) + sum(
                    sol_state[2][i,t] * sol_dual_var_start_up[i,t] for
                    i in 1:N1 for t in 1:T) + sum(
                    sol_state[3][i,t] * sol_dual_var_start_down[i,t] for
                    i in 1:N1 for t in 1:T) + sum(
                    worst_case[t] * sol_dual_uncertainty[t] for t in 1:T)
                cstr=@build_constraint(lagrangian.theta <= intercept - sum(
                    sol_state[1][i,t] * dual_var_is_on[i,t] for
                    i in 1:N1 for t in 0:T) - sum(
                    sol_state[2][i,t] * dual_var_start_up[i,t] for
                    i in 1:N1 for t in 1:T) - sum(
                    sol_state[3][i,t] * dual_var_start_down[i,t] for
                    i in 1:N1 for t in 1:T) - sum(
                    worst_case[t] * dual_var_uncertainty[t] for t in 1:T))
                MOI.submit(lagrangian.model, MOI.LazyConstraint(cb_data), cstr)
                cut = cutRO(intercept, sol_state[1], sol_state[2], sol_state[3], worst_case, sol_dual_var_is_on, sol_dual_var_start_up, sol_dual_var_start_down, sol_dual_uncertainty)
                push!(list_cuts, cut)

            end

            if current_obj + obj - theta_val>= 0.999999*LB
                LB=current_obj + obj - theta_val
                # println("LB : ", LB)
                if k>=10 #Force to the new best solution after 10 iterations
                    vars= vcat(lagrangian.vars,
                        [lagrangian.theta])
                    
                    vals= vcat(vars_val,
                        [obj])
                    
                    MOI.submit(lagrangian.model, MOI.HeuristicSolution(cb_data), vars, vals)
                end
            end
        end
        return
    end

    set_optimizer_attribute(lagrangian.model, "LazyConstraints", 1)
    MOI.set(lagrangian.model, Gurobi.CallbackFunction(), my_callback_function)

    JuMP.optimize!(lagrangian.model)

    worst_case_cost_bound = JuMP.objective_bound(lagrangian.model)

    worst_case_cost_obj = JuMP.objective_value(lagrangian.model)

    worst_case = Dict(name => round(JuMP.value(var)) for (name, var) in lagrangian.uncertainty)

    sol_dual_var_is_on = Dict((i,t) => JuMP.value(dual_var_is_on[i,t]) for i in 1:N1 for t in 0:T)
    sol_dual_var_start_up = Dict((i,t) => JuMP.value(dual_var_start_up[i,t]) for i in 1:N1 for t in 1:T)
    sol_dual_var_start_down = Dict((i,t) => JuMP.value(dual_var_start_down[i,t]) for i in 1:N1 for t in 1:T)

    sol_dual_var = [sol_dual_var_is_on, sol_dual_var_start_up, sol_dual_var_start_down]

    sol_dual_uncertainty = JuMP.value.(lagrangian.dual_var_uncertainty)

    theta_val = JuMP.value(lagrangian.theta)

    dual_demand = JuMP.value.(lagrangian.dual_demand)

    JuMP.set_objective_function(lagrangian.model, primal_obj)

    for cut in list_cuts
        cstr = @constraint(lagrangian.model, lagrangian.theta <= cut.intercept - sum(
            cut.is_on[i,t] * dual_var_is_on[i,t] for
            i in 1:N1 for t in 0:T) - sum(
            cut.start_up[i,t] * dual_var_start_up[i,t] for
            i in 1:N1 for t in 1:T) - sum(
            cut.start_down[i,t] * dual_var_start_down[i,t] for
            i in 1:N1 for t in 1:T) - sum(
            cut.uncertainty[t] * dual_var_uncertainty[t] for t in 1:T))
    end

    return list_cuts

    return worst_case_cost_obj, worst_case_cost_bound, worst_case, theta_val, sol_dual_var, sol_dual_uncertainty, dual_demand
end

function get_worst_case_RO_lagrangianUncertainty_callback(twoROmodel::twoRO, states_1::Dict{Symbol, Float64}, states_2::Dict{Symbol, Float64})
    """
    Get worst-case cost from Lagrangian relaxation
    """
    lagrangian = twoROmodel.lagrangian

    cstr = lagrangian.upper_constraint

    for (name, var) in lagrangian.dual_var_states_1
        set_normalized_coefficient(cstr, var, -states_1[name])
    end

    for (name, var) in lagrangian.dual_var_states_2
        set_normalized_coefficient(cstr, var, -states_2[name])
    end

    LB=0.0
    k = 0

    primal_obj = JuMP.objective_function(lagrangian.model)

    start = time()
    function my_callback_function(cb_data, cb_where::Cint)

        if cb_where==GRB_CB_MIPSOL
            k+=1
    
            Gurobi.load_callback_variable_primal(cb_data, cb_where)

            sol_dual_uncertainty = Dict(name => callback_value(cb_data, var) for (name, var) in lagrangian.dual_var_uncertainty)

            theta_val = callback_value(cb_data, lagrangian.theta)

            vars_val = callback_value.(cb_data, lagrangian.vars)

            current_obj = callback_value(cb_data, primal_obj)

            result = solve_second_stage_RO_lagrangianUncertainty(twoROmodel, states_1, sol_dual_uncertainty)
            obj = result.objective
            worst_case = result.worst_case

            if obj<= 0.999999*theta_val

                intercept = obj + sum(
                    worst_case[name] * sol_dual_uncertainty[name] for  (name, _) in lagrangian.dual_var_uncertainty)

                cstr=@build_constraint(lagrangian.theta <= intercept - sum(
                    worst_case[name] * var for  (name, var) in lagrangian.dual_var_uncertainty))
                MOI.submit(lagrangian.model, MOI.LazyConstraint(cb_data), cstr)
            end

            if current_obj + obj - theta_val>= 0.999999*LB
                LB=current_obj + obj - theta_val
                if k>=10 #Force to the new best solution after 10 iterations
                    vars= vcat(lagrangian.vars,
                        [lagrangian.theta])
                    
                    vals= vcat(vars_val,
                        [obj])
                    
                    MOI.submit(lagrangian.model, MOI.HeuristicSolution(cb_data), vars, vals)
                end
            end
        end
        return
    end

    set_optimizer_attribute(lagrangian.model, "LazyConstraints", 1)
    MOI.set(lagrangian.model, Gurobi.CallbackFunction(), my_callback_function)

    JuMP.optimize!(lagrangian.model)

    computation_time = time() - start
    # println("Time worst-case Lagrangian Uncertainty callback: ", computation_time)
    worst_case_cost_obj = JuMP.objective_value(lagrangian.model)

    worst_case = Dict(name => round(JuMP.value(var)) for (name, var) in lagrangian.uncertainty)

    return (objective = worst_case_cost_obj, worst_case = worst_case)
end

function get_price_demand_RO(twoROmodel::twoRO, states_1_val::Dict{Symbol, Float64}, solution_uncertainty::Dict{Symbol, Float64}, instance::Instance)
    """
    Get worst-case cost from Lagrangian relaxation
    """
    subproblem = twoROmodel.subproblem

    states_1 = subproblem.states_1
    states_2 = subproblem.states_2
    uncertainty = subproblem.uncertainty

    T= instance.TimeHorizon
    Next=instance.Next
    Buses=1:size(Next)[1]

    for (name, var) in states_1
        fix(var, states_1_val[name]; force=true)
    end

    for (name, var) in uncertainty
        fix(var, solution_uncertainty[name]; force=true)
    end

    JuMP.optimize!(subproblem.model)

    states_2_val = Dict{Symbol, Float64}(name => round(JuMP.value(var)) for (name, var) in states_2)

    for (name, var) in states_2
        fix(var, states_2_val[name]; force=true)
    end

    undo_relax = JuMP.relax_integrality(subproblem.model)

    JuMP.optimize!(subproblem.model)

    obj = JuMP.objective_value(subproblem.model)
    bound = JuMP.objective_bound(subproblem.model)

    price_demand = Matrix{Float64}(undef, T, length(Buses))
    for t in 1:T
        for b in Buses
            price_demand[t,b] = JuMP.dual(subproblem.cstr_demand[t,b])
        end
    end

    undo_relax()

    for (_, var) in states_1
        JuMP.unfix(var)
    end
    
    for (_, var) in states_2
        JuMP.unfix(var)
    end

    for (_, var) in uncertainty
        JuMP.unfix(var)
    end

    return price_demand
end

function solve_price_relaxation(twoROmodel::twoRO, instance::Instance, price_demand::Matrix{Float64}, states_1_val::Dict{Symbol, Float64}, worst_case::Dict{Symbol, Float64})
    """
    Get worst-case cost from Lagrangian relaxation
    """

    T= instance.TimeHorizon
    N1 = instance.N1
    N2 = instance.N - N1
    thermal_units=instance.Thermalunits
    thermal_units1 = thermal_units[1:N1]
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)

    relaxation1 = twoROmodel.priceRelaxation.model1
    model1 = relaxation1.model

    muup = relaxation1.muup
    mudown = relaxation1.mudown
    states_1 = relaxation1.states
    power1 = relaxation1.power
    power_shedding1 = relaxation1.power_shedding
    power_curtailement1 = relaxation1.power_curtailement
    flow1 = relaxation1.flow

    for (name, var) in states_1
        fix(var, states_1_val[name]; force=true)
    end

    primal_obj = JuMP.objective_function(model1)

    new_obj_function = @expression(model1, primal_obj - sum(price_demand[t, unit.Bus] * power1[unit.name, t] for unit in thermal_units1 for t in 1:T)-sum(price_demand[t,b] * (power_shedding1[b,t] - power_curtailement1[b,t]) for b in Buses for t in 1:T)-sum(price_demand[t,b] * flow1[(line.id, t)] for line in Lines for t in 1:T for b in Buses if line.b2==b) + sum(price_demand[t,b] * flow1[(line.id, t)] for line in Lines for t in 1:T for b in Buses if line.b1==b))

    JuMP.set_objective_function(
        model1,
        new_obj_function
    )

    JuMP.optimize!(model1)

    gradient1 = Matrix{Float64}(undef, T, length(Buses))
    for t in 1:T
        for b in Buses
            gradient1[t,b] = sum(JuMP.value(power1[i,t]) for i in 1:N1 if thermal_units[i].Bus==b; init = 0) + JuMP.value(power_shedding1[b,t]) - JuMP.value(power_curtailement1[b,t]) + sum(JuMP.value(flow1[(line.id, t)]) for line in Lines if line.b2==b; init = 0) - sum(JuMP.value(flow1[(line.id, t)]) for line in Lines if line.b1==b; init = 0)
        end
    end

    dual_muup = JuMP.dual.(muup)
    dual_mudown = JuMP.dual.(mudown)
    
    obj1 = JuMP.objective_value(model1)

    for (_, var) in states_1
        JuMP.unfix(var)
    end

    JuMP.set_objective_function(model1, primal_obj)

    relaxation2 = twoROmodel.priceRelaxation.model2
    model2 = relaxation2.model
    power2 = relaxation2.power
    primal_obj2 = JuMP.objective_function(model2)
    new_obj_function2 = @expression(model2, primal_obj2 - sum(price_demand[t, thermal_units[i+N1].Bus] * power2[i, t] for i in 1:N2 for t in 1:T))
    JuMP.set_objective_function(
        model2,
        new_obj_function2
    )
    JuMP.optimize!(model2)
    obj2 = JuMP.objective_value(model2)

    gradient2 = Matrix{Float64}(undef, T, length(Buses))
    for t in 1:T
        for b in Buses
            gradient2[t,b] = sum(JuMP.value(power2[i,t]) for i in 1:N2 if thermal_units[i+N1].Bus==b; init = 0)
        end
    end

    JuMP.set_objective_function(model2, primal_obj2)

    intercept = sum(price_demand[t,b] * instance.Demandbus[b][t]*(1+1.96*0.025*worst_case[Symbol("uncertainty[$t]")]) for b in Buses for t in 1:T)

    gradient = - gradient1 - gradient2

    return ResultsPriceRelaxation(obj1+obj2+intercept, dual_muup, dual_mudown, price_demand, gradient)
end

function get_SB_cut_RO(twoROmodel::twoRO, states_1_val::Dict{Symbol, Float64}, worst_case::Dict{Symbol, Float64})
    """
    Get worst-case cost from Lagrangian relaxation
    """
    subproblem = twoROmodel.subproblem

    states_1 = subproblem.states_1
    uncertainty = subproblem.uncertainty

    for (name, var) in states_1
        fix(var, states_1_val[name]; force=true)
    end

    for (name, var) in uncertainty
        fix(var, worst_case[name]; force=true)
    end

    undo_relax = JuMP.relax_integrality(subproblem.model)

    JuMP.optimize!(subproblem.model)

    sol_dual_var_states_1 = Dict(name => JuMP.dual(JuMP.FixRef(var)) for (name, var) in states_1)

    undo_relax()

    for (_, var) in states_1
        JuMP.unfix(var)
    end

    primal_obj = JuMP.objective_function(subproblem.model)

    JuMP.set_objective_function(
        subproblem.model,
        @expression(subproblem.model, primal_obj + sum(
            sol_dual_var_states_1[name] * (states_1_val[name] - var) for
            (name, var) in states_1))
    )

    JuMP.optimize!(subproblem.model)

    gradient = Dict{Symbol, Float64}(name => states_1_val[name] - JuMP.value(var) for (name, var) in states_1)

    bound = JuMP.objective_bound(subproblem.model)

    for (_, var) in uncertainty
        JuMP.unfix(var)
    end

    JuMP.set_objective_function(subproblem.model, primal_obj)

    return (objective = bound, sol_dual_var_states_1 = sol_dual_var_states_1, gradient = gradient)
end

function get_optimal_price_demand(twoROmodel::twoRO, states_1::Dict{Symbol, Float64}, states_2::Dict{Symbol, Float64}, worst_case::Dict{Symbol, Float64}, instance::Instance)
    """
    Get worst-case cost from Lagrangian relaxation
    """

    T = instance.TimeHorizon
    Buses = 1:size(instance.Next)[1]

    priceproblem = twoROmodel.priceRelaxation.price_problem

    dual_demand = priceproblem.dual_demand

    cstr = priceproblem.upper_constraint

    for (name, var) in priceproblem.dual_var_states_1
        set_normalized_coefficient(cstr, var, -states_1[name])
    end

    for (name, var) in priceproblem.dual_var_states_2
        set_normalized_coefficient(cstr, var, -states_2[name])
    end

    LB = 0.0
    UB = 1e9
    k = 0

    primal_obj = JuMP.objective_function(priceproblem.model)

    new_obj_function = @expression(priceproblem.model, primal_obj + sum(dual_demand[t,b] * instance.Demandbus[b][t]*(1+1.96*0.025*worst_case[Symbol("uncertainty[$t]")]) for b in Buses for t in 1:T))

    JuMP.set_objective_function(
        priceproblem.model,
        new_obj_function
    )

    constraints = ConstraintRef[]

    while 100*(UB-LB)/UB > 0.1 && k <= 50
        k += 1

        JuMP.optimize!(priceproblem.model)

        current_obj = JuMP.objective_value(priceproblem.model)

        UB = current_obj

        price_demand = JuMP.value.(dual_demand)

        results_relaxation = solve_price_relaxation(twoROmodel, instance, price_demand, states_1, worst_case)

        LB = max(LB, results_relaxation.obj)

        intercept = sum(price_demand[t,b] * instance.Demandbus[b][t]*(1+1.96*0.025*worst_case[Symbol("uncertainty[$t]")]) for b in Buses for t in 1:T)
        
        if 100*(UB-LB)/UB > 0.1 && k <= 50
            cstr = @constraint(priceproblem.model, priceproblem.theta <= results_relaxation.obj - intercept + sum(results_relaxation.gradient[t,b] * (dual_demand[t,b] - price_demand[t,b]) for b in Buses for t in 1:T))
            push!(constraints, cstr)
        else
            JuMP.set_objective_function(priceproblem.model, primal_obj)
            for cstr in constraints
                delete(priceproblem.model, cstr)
            end
            return results_relaxation
        end
    end

    price_demand = JuMP.value.(dual_demand)

    JuMP.set_objective_function(priceproblem.model, primal_obj)

    for cstr in constraints
        delete(priceproblem.model, cstr)
    end

    return (LB = LB, UB = UB, price_demand = price_demand)
end

function get_lagrangian_cut(twoROmodel::twoRO, states_1::Dict{Symbol, Float64}, states_2::Dict{Symbol, Float64}, worst_case::Dict{Symbol, Float64})
    """
    Get worst-case cost from Lagrangian relaxation
    """

    lagrangian = twoROmodel.lagrangianStates

    cstr = lagrangian.upper_constraint

    for (name, var) in lagrangian.dual_var_states_2
        set_normalized_coefficient(cstr, var, -states_2[name])
    end

    for (name, var) in lagrangian.dual_var_uncertainty
        set_normalized_coefficient(cstr, var, -worst_case[name])
    end

    primal_obj = JuMP.objective_function(lagrangian.model)

    JuMP.set_objective_function(
        lagrangian.model,
        @expression(lagrangian.model, primal_obj + sum(
            var * states_1[name] for
            (name, var) in lagrangian.dual_var_states_1))
    )

    LB = 0.0
    UB = 1e9
    k = 0

    dual_states_1_prev = Dict(name => 0.0 for (name, var) in lagrangian.dual_var_states_1)
    best_dual_states_1 = Dict(name => 0.0 for (name, var) in lagrangian.dual_var_states_1)

    constraints = ConstraintRef[]
    MaxIter = 5
    while 100*(UB-LB)/UB > 0.1 && k <= MaxIter
        # println((k, UB, LB, 100*(UB-LB)/UB))
        k += 1

        JuMP.optimize!(lagrangian.model)

        current_obj = JuMP.objective_value(lagrangian.model)

        UB = current_obj

        dual_states_1 = Dict(name => JuMP.value(var) for (name, var) in lagrangian.dual_var_states_1)
        theta_val = JuMP.value(lagrangian.theta)

        results_subproblem = solve_second_stage_RO_lagrangianStates(twoROmodel, states_1, dual_states_1, worst_case)

        if current_obj - theta_val + results_subproblem.bound > LB
            LB = current_obj - theta_val + results_subproblem.bound
            best_dual_states_1 = dual_states_1
        end
        # println(sum(abs(dual_states_1[name]-dual_states_1_prev[name]) for (name, _) in lagrangian.dual_var_states_1))

        # println(sum(abs(results_subproblem.states_1[name]-states_1[name]) for (name, _) in states_1))
        if 100*(UB-LB)/UB > 0.1 && k <= MaxIter && sum(abs(dual_states_1[name]-dual_states_1_prev[name]) for (name, _) in lagrangian.dual_var_states_1) > 1e-6
            dual_states_1_prev = dual_states_1
            cstr = @constraint(lagrangian.model, lagrangian.theta <= results_subproblem.objective + sum(
                results_subproblem.states_1[name] * (dual_states_1[name] - lagrangian.dual_var_states_1[name]) for
                (name, var) in lagrangian.dual_var_states_1))
            push!(constraints, cstr)
        else
            JuMP.set_objective_function(lagrangian.model, primal_obj)
            for cstr in constraints
                delete(lagrangian.model, cstr)
            end
            return (objective = LB, bound = UB, iteration = k, sol_dual_var_states_1 = dual_states_1)
        end
    end
end


function compute_lagrangian_gradient(twoROmodel::twoRO, states_1_val::Dict{Symbol, Float64}, worst_case::Dict{Symbol, Float64}, sol_dual_var_states_1::Dict{Symbol, Float64})
    """
    Get worst-case cost from Lagrangian relaxation
    """
    subproblem = twoROmodel.subproblem

    # unset_silent(subproblem.model)

    states_1 = subproblem.states_1
    uncertainty = subproblem.uncertainty

    for (name, var) in uncertainty
        fix(var, worst_case[name]; force=true)
    end

    primal_obj = JuMP.objective_function(subproblem.model)

    JuMP.set_objective_function(
        subproblem.model,
        @expression(subproblem.model, primal_obj + sum(
            sol_dual_var_states_1[name] * (states_1_val[name] - var) for
            (name, var) in states_1))
    )

    start = time()
    JuMP.optimize!(subproblem.model)
    # println(time() - start)

    # println(JuMP.value(primal_obj))

    gradient = Dict{Symbol, Float64}(name => states_1_val[name] - JuMP.value(var) for (name, var) in states_1)

    obj = JuMP.objective_value(subproblem.model)
    bound = JuMP.objective_bound(subproblem.model)

    for (_, var) in uncertainty
        JuMP.unfix(var)
    end

    JuMP.set_objective_function(subproblem.model, primal_obj)

    return (objective = obj, bound=bound, sol_dual_var_states_1 = sol_dual_var_states_1, gradient = gradient)
end

function get_lagrangian_cut2(twoROmodel::twoRO, states_1::Dict{Symbol, Float64}, worst_case::Dict{Symbol, Float64}, bound::Float64)
    resL = RDDIP.get_SB_cut_RO(twoROmodel, states_1, worst_case)
    dual = resL.sol_dual_var_states_1
    n = sqrt(sum(g^2 for (name, g) in resL.gradient))
    k=0
    while n^2>=1 && resL.objective <= bound && k<=5
        k+=1
        α = (bound-resL.objective)/n^2
        for (name, g) in resL.gradient
            dual[name] = dual[name] + α*g/n
        end
        resL = RDDIP.compute_lagrangian_gradient(twoROmodel, states_1, worst_case, dual)
        n = sqrt(sum(g^2 for (name, g) in resL.gradient))
        # println((k,n, resL.objective))
    end
    # println((k,n, resL.objective))
    return resL
end

function get_extended_cut_RO(twoROmodel::twoRO, sol_dual_var_states_1::Dict{Symbol, Float64}, states_1_val::Dict{Symbol, Float64}, worst_case::Dict{Symbol, Float64})
    
    subproblem = twoROmodel.subproblemextended
    states_1 = subproblem.states_1
    uncertainty = subproblem.uncertainty
    
    for (name, var) in uncertainty
        fix(var, worst_case[name]; force=true)
    end
    
    primal_obj = JuMP.objective_function(subproblem.model)

    JuMP.set_objective_function(
        subproblem.model,
        @expression(subproblem.model, primal_obj + sum(
            sol_dual_var_states_1[name] * (states_1_val[name] - var) for
            (name, var) in states_1))
    )

    JuMP.optimize!(subproblem.model)

    gradient = Dict{Symbol, Float64}(name => states_1_val[name] - JuMP.value(var) for (name, var) in states_1)

    bound = JuMP.objective_bound(subproblem.model)
    obj = JuMP.objective_value(subproblem.model)

    for (_, var) in uncertainty
        JuMP.unfix(var)
    end

    JuMP.set_objective_function(subproblem.model, primal_obj)

    return (bound = bound, objective = obj, gradient = gradient)
end