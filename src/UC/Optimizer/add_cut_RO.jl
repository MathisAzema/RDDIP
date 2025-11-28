mutable struct ResultsPriceRelaxation
    obj::Float64
    dual_muup::Matrix{Float64}
    dual_mudown::Matrix{Float64}
    price_demand::Matrix{Float64}
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

    println(price[:, 1])

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

function get_worst_case_RO_continuous(twoROmodel::twoRO, solution_xN1::Vector{Matrix{Float64}}, instance::Instance)
    """
    Get worst-case cost from Lagrangian relaxation
    """
    oraclecontinuous = twoROmodel.oracleContinuousRO

    dual_var_is_on = oraclecontinuous.dual_var_is_on
    dual_var_start_up = oraclecontinuous.dual_var_start_up
    dual_var_start_down = oraclecontinuous.dual_var_start_down

    T= instance.TimeHorizon
    N1 = instance.N1
    N2 = instance.N - N1

    solution_is_on = solution_xN1[1]
    solution_start_up = solution_xN1[2]
    solution_start_down = solution_xN1[3]

    primal_obj = JuMP.objective_function(oraclecontinuous.model)

    JuMP.set_objective_function(
        oraclecontinuous.model,
        @expression(oraclecontinuous.model, primal_obj + sum(
            dual_var_is_on[i,t] * solution_is_on[i,t+1] for
            i in 1:N1 for t in 0:T)) + sum(
            dual_var_start_up[i,t] * solution_start_up[i,t] for
            i in 1:N1 for t in 1:T) + sum(
            dual_var_start_down[i,t] * solution_start_down[i,t] for
            i in 1:N1 for t in 1:T)
    )

    JuMP.optimize!(oraclecontinuous.model)

    worst_case = Dict(t => round(JuMP.value(oraclecontinuous.uncertainty[t])) for t in 1:T)

    JuMP.set_objective_function(oraclecontinuous.model, primal_obj)

    return worst_case
end

function solve_second_stage_RO(twoROmodel::twoRO, solution_xN1::Vector{Matrix{Float64}}, solution_uncertainty::Dict{Int64, Float64}, instance::Instance)
    """
    Get worst-case cost from Lagrangian relaxation
    """
    subproblem = twoROmodel.subproblem

    is_on = subproblem.is_on
    uncertainty = subproblem.uncertainty

    T= instance.TimeHorizon
    N = instance.N
    N1 = instance.N1
    N2 = instance.N - N1

    for i in 1:N1
        for t in 0:T
            fix(is_on[i,t], solution_xN1[1][i,t+1]; force=true)
        end
        for t in 1:T
            fix(subproblem.start_up[i,t], solution_xN1[2][i,t]; force=true)
            fix(subproblem.start_down[i,t], solution_xN1[3][i,t]; force=true)
        end
    end

    for t in 1:T
        fix(uncertainty[t], solution_uncertainty[t]; force=true)
    end

    JuMP.optimize!(subproblem.model)


    obj = JuMP.objective_value(subproblem.model)
    bound = JuMP.objective_bound(subproblem.model)
    
    solution_is_on = Matrix{Float64}(undef, N, T+1)
    solution_start_up = Matrix{Float64}(undef, N, T)
    solution_start_down = Matrix{Float64}(undef, N, T)
    for i in 1:N
        for t in 0:T
            solution_is_on[i,t+1]=round(JuMP.value(is_on[i,t]))
        end
    end
    for i in 1:N
        for t in 1:T
            solution_start_up[i,t]=round(JuMP.value(subproblem.start_up[i,t]))
            solution_start_down[i,t]=round(JuMP.value(subproblem.start_down[i,t]))
            if i<=N1 && abs(solution_start_up[i,t] - solution_xN1[2][i,t])>0.1
                println("Error start up ", (i,t))
            end
            if i<=N1 && abs(solution_start_down[i,t] - solution_xN1[3][i,t])>0.1
                println("Error start down ", (i,t))
            end
        end
    end

    solution_xN2 = [solution_is_on[N1+1:end,:], solution_start_up[N1+1:end,:], solution_start_down[N1+1:end,:]]

    for i in 1:N1
        for t in 0:T
            JuMP.unfix(is_on[i,t])
        end
        for t in 1:T
            JuMP.unfix(subproblem.start_up[i,t])
            JuMP.unfix(subproblem.start_down[i,t])
        end
    end

    for t in 1:T
        JuMP.unfix(uncertainty[t])
    end

    return obj, bound, solution_xN2
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

function solve_second_stage_RO_lagrangianUncertainty(twoROmodel::twoRO, instance::Instance, solution_xN1::Vector{Matrix{Float64}}, dual_uncertainty::Vector{Float64})
    """
    Get worst-case cost from Lagrangian relaxation
    """
    subproblem = twoROmodel.subproblem

    is_on = subproblem.is_on
    uncertainty = subproblem.uncertainty

    T= instance.TimeHorizon
    N = instance.N
    N1 = instance.N1
    N2 = instance.N - N1

    solution_is_on = solution_xN1[1]

    for i in 1:N1
        for t in 0:T
            fix(is_on[i,t], solution_is_on[i,t+1]; force=true)
        end
    end

    primal_obj = JuMP.objective_function(subproblem.model)

    JuMP.set_objective_function(
        subproblem.model,
        @expression(subproblem.model, primal_obj - sum(
            dual_uncertainty[t] * uncertainty[t] for t in 1:T))
    )

    JuMP.optimize!(subproblem.model)

    obj = JuMP.objective_value(subproblem.model)

    worst_case = [round(JuMP.value(uncertainty[t])) for t in 1:T]

    JuMP.set_objective_function(subproblem.model, primal_obj)
    
    for i in 1:N1
        for t in 0:T
            JuMP.unfix(is_on[i,t])
        end
    end


    return obj, worst_case
end

function add_cut_RO_bin(cb_data, twoROmodel::twoRO, instance::Instance, Time_subproblem, solution_xN1::Vector{Matrix{Float64}}, solution_gamma::Dict{Tuple{Int64, Int64, Int64}, Float64}; gap)
    """
    Add Benders' cut
    """
    # println("Add cut RO bin")
    T= instance.TimeHorizon
    thermal_units=instance.Thermalunits
    N = instance.N
    N1 = instance.N1
    thermal_units_N1=thermal_units[1:N1]    

    master_pb = twoROmodel.master_pb

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


    worst_case_continuous = get_worst_case_RO_continuous(twoROmodel, solution_xN1, instance)

    obj_continuous, bound, solution_xN2 = solve_second_stage_RO(twoROmodel, solution_xN1, worst_case_continuous, instance)

    add_cut = false

    update_UB = false

    if thermal_fixed_cost_val+obj_continuous >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
        worst_case = worst_case_continuous
        
        obj, bound, price_demand = get_price_demand_RO(twoROmodel, solution_xN1, worst_case, instance)

        results_price_relaxation = solve_price_relaxation(twoROmodel, instance, price_demand, solution_xN1, worst_case)

        worst_case_cost_obj = results_price_relaxation.obj

        if thermal_fixed_cost_val+results_price_relaxation.obj >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
            _add_optimality_cuts_RO_bin(cb_data, master_pb, results_price_relaxation, instance, solution_gamma, current_intervals)
            add_cut = true
        end

        # println(("heur", add_cut, thermal_fixed_cost_val, thermal_fixed_cost_val + results_price_relaxation.obj, thermal_fixed_cost_val + obj_continuous, thermal_fixed_cost_val+thermal_cost_val))

    end

    if !add_cut
        update_UB = true
        # println("Compute worst-case Lagrangian Uncertainty ", thermal_fixed_cost_val + thermal_cost_val, " ", thermal_fixed_cost_val+obj_continuous)
        worst_case_cost_obj, worst_case = get_worst_case_RO_lagrangianUncertainty_callback(twoROmodel, solution_xN1, solution_xN2, instance)

        obj, bound, price_demand = get_price_demand_RO(twoROmodel, solution_xN1, worst_case, instance)

        results_price_relaxation = solve_price_relaxation(twoROmodel, instance, price_demand, solution_xN1, worst_case)

        if thermal_fixed_cost_val+results_price_relaxation.obj >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
            _add_optimality_cuts_RO_bin(cb_data, master_pb, results_price_relaxation, instance, solution_gamma, current_intervals)
            add_cut = true
        end

        # println(("callback", add_cut, thermal_fixed_cost_val, thermal_fixed_cost_val + results_price_relaxation.obj, thermal_fixed_cost_val + worst_case_cost_obj, thermal_fixed_cost_val+thermal_cost_val))
        # println(("callback2", add_cut, thermal_fixed_cost_val, results_price_relaxation.obj, worst_case_cost_obj, thermal_cost_val))

        #Calcul des coefficients de la coupe

        cost_SB, sol_dual_var = get_SB_cut_RO(twoROmodel, solution_xN1, worst_case, instance)

        # println((thermal_fixed_cost_val, worst_case_cost_obj, cost_SB, thermal_cost_val))
        if thermal_fixed_cost_val+cost_SB >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
            _add_SB_cut(cb_data, master_pb, instance, solution_xN1, cost_SB, sol_dual_var)
            add_cut = true
        end
    end

    return update_UB, thermal_fixed_cost_val+worst_case_cost_obj, worst_case_cost_obj
    
end

function add_cut_RO_bin2(cb_data, twoROmodel::twoRO, instance::Instance, Time_subproblem, solution_xN1::Vector{Matrix{Float64}}, solution_gamma::Dict{Tuple{Int64, Int64, Int64}, Float64}, Γ; gap)
    """
    Add Benders' cut
    """
    # println("Add cut RO bin")
    T= instance.TimeHorizon
    thermal_units=instance.Thermalunits
    N = instance.N
    N1 = instance.N1
    thermal_units_N1=thermal_units[1:N1]    

    master_pb = twoROmodel.master_pb

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


    worst_case_continuous = get_worst_case_RO_continuous(twoROmodel, solution_xN1, instance)

    obj_continuous, bound, solution_xN2 = solve_second_stage_RO(twoROmodel, solution_xN1, worst_case_continuous, instance)

    add_cut = false

    update_UB = false

    if thermal_fixed_cost_val+obj_continuous >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
        worst_case = worst_case_continuous
        
        obj, bound, price_demand = get_price_demand_RO(twoROmodel, solution_xN1, worst_case, instance)

        results_price_relaxation = solve_price_relaxation(twoROmodel, instance, price_demand, solution_xN1, worst_case)

        worst_case_cost_obj = results_price_relaxation.obj

        if thermal_fixed_cost_val+results_price_relaxation.obj >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
            _add_optimality_cuts_RO_bin(cb_data, master_pb, results_price_relaxation, instance, solution_gamma, current_intervals)
            add_cut = true
        end

        # println(("heur", add_cut, thermal_fixed_cost_val, thermal_fixed_cost_val + results_price_relaxation.obj, thermal_fixed_cost_val + obj_continuous, thermal_fixed_cost_val+thermal_cost_val))

    end

    if !add_cut
        update_UB = true
        # println("Compute worst-case Lagrangian Uncertainty ", thermal_fixed_cost_val + thermal_cost_val, " ", thermal_fixed_cost_val+obj_continuous)
        worst_case_cost_obj, worst_case = get_worst_case_RO_enumeration(twoROmodel, solution_xN1, instance, Γ)

        obj, bound, price_demand = get_price_demand_RO(twoROmodel, solution_xN1, worst_case, instance)

        results_price_relaxation = solve_price_relaxation(twoROmodel, instance, price_demand, solution_xN1, worst_case)

        if thermal_fixed_cost_val+results_price_relaxation.obj >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
            _add_optimality_cuts_RO_bin(cb_data, master_pb, results_price_relaxation, instance, solution_gamma, current_intervals)
            add_cut = true
        end

        # println(("callback", add_cut, thermal_fixed_cost_val, thermal_fixed_cost_val + results_price_relaxation.obj, thermal_fixed_cost_val + worst_case_cost_obj, thermal_fixed_cost_val+thermal_cost_val))
        # println(("callback2", add_cut, thermal_fixed_cost_val, results_price_relaxation.obj, worst_case_cost_obj, thermal_cost_val))

        #Calcul des coefficients de la coupe

        cost_SB, sol_dual_var = get_SB_cut_RO(twoROmodel, solution_xN1, worst_case, instance)

        # println((thermal_fixed_cost_val, worst_case_cost_obj, cost_SB, thermal_cost_val))
        if thermal_fixed_cost_val+cost_SB >= (1+0.01*gap/100)*(thermal_fixed_cost_val+thermal_cost_val)
            _add_SB_cut(cb_data, master_pb, instance, solution_xN1, cost_SB, sol_dual_var)
            add_cut = true
        end
    end

    return update_UB, thermal_fixed_cost_val+worst_case_cost_obj, worst_case_cost_obj
    
end

function get_worst_case_RO_enumeration(twoROmodel::twoRO, solution_xN1::Vector{Matrix{Float64}}, instance::Instance, Γ::Int64)
    T = instance.TimeHorizon
    uncertainty_set = generate_all_Γ_tuple(T, Γ)
    worst_case_cost = -Inf
    worst_case = Dict{Int, Float64}()
    for uncertainty in uncertainty_set
        uncertainty_dict = Dict(t => uncertainty[t]*1.0 for t in 1:T)
        obj, bound, _ = solve_second_stage_RO(twoROmodel, solution_xN1, uncertainty_dict, instance)
        if obj > worst_case_cost
            worst_case_cost = obj
            worst_case = uncertainty_dict
        end
    end
    return worst_case_cost, worst_case
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

function _add_SB_cut(cb_data, master_pb, instance::Instance, solution_xN1::Vector{Matrix{Float64}}, cost_SB::Float64, sol_dual_var::Vector{Dict{Tuple{Int64, Int64}, Float64}})

    T= instance.TimeHorizon
    N1 = instance.N1
    is_on = master_pb[:is_on]
    start_up = master_pb[:start_up]
    start_down = master_pb[:start_down]
    thermal_cost=master_pb[:thermal_cost]

    cstr=@build_constraint(cost_SB + sum(sol_dual_var[1][i,t] * (is_on[i,t] - solution_xN1[1][i,t+1]) + sol_dual_var[2][i,t] * (start_up[i,t] - solution_xN1[2][i,t]) + sol_dual_var[3][i,t] * (start_down[i,t] - solution_xN1[3][i,t])  for i in 1:N1 for t in 1:T) <= thermal_cost) #attention 0 ou 1:T ?
    MOI.submit(master_pb, MOI.LazyConstraint(cb_data), cstr)
    # println("add_lagrangian_cut :", cost_SB)
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

            tm= [t for t in 1:T if worst_case[t]>0.1]


            println((tm, current_obj, current_obj + obj - theta_val, obj, theta_val))

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
                println("LB : ", LB)
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

function get_worst_case_RO_lagrangianUncertainty_callback(twoROmodel::twoRO, solution_xN1::Vector{Matrix{Float64}}, solution_xN2::Vector{Matrix{Float64}}, instance::Instance)
    """
    Get worst-case cost from Lagrangian relaxation
    """
    lagrangian = twoROmodel.lagrangianUncertainty

    dual_var_uncertainty = lagrangian.dual_var_uncertainty

    T= instance.TimeHorizon
    N1 = instance.N1
    N2 = instance.N - N1

    solution_is_on = solution_xN1[1]
    solution_start_up = solution_xN1[2]
    solution_start_down = solution_xN1[3]

    solution_is_on_N2 = solution_xN2[1]
    solution_start_up_N2 = solution_xN2[2]
    solution_start_down_N2 = solution_xN2[3]

    cstr = lagrangian.upper_constraint
    
    for i in 1:N1
        for t in 0:T
            value = solution_is_on[i,t+1]
            var = lagrangian.dual_var_is_on[i,t]
            set_normalized_coefficient(cstr, var, -value)
        end
    end

    for i in 1:N1
        for t in 1:T
            value = solution_start_up[i,t]
            var = lagrangian.dual_var_start_up[i,t]
            set_normalized_coefficient(cstr, var, -value)
            value = solution_start_down[i,t]
            var = lagrangian.dual_var_start_down[i,t]
            set_normalized_coefficient(cstr, var, -value)
        end
    end


    for k in 1:N2
        i = N1 + k
        for t in 0:T
            value = solution_is_on_N2[k,t+1]
            var = lagrangian.dual_var_is_on[i,t]
            set_normalized_coefficient(cstr, var, -value)
        end
    end

    for k in 1:N2
        i = N1 + k
        for t in 1:T
            value = solution_start_up_N2[k,t]
            var = lagrangian.dual_var_start_up[i,t]
            set_normalized_coefficient(cstr, var, -value)
            value = solution_start_down_N2[k,t]
            var = lagrangian.dual_var_start_down[i,t]
            set_normalized_coefficient(cstr, var, -value)
        end
    end

    LB=0.0
    k = 0

    primal_obj = JuMP.objective_function(lagrangian.model)

    function my_callback_function(cb_data, cb_where::Cint)

        if cb_where==GRB_CB_MIPSOL
            k+=1
    
            Gurobi.load_callback_variable_primal(cb_data, cb_where)

            sol_dual_uncertainty = callback_value.(cb_data, dual_var_uncertainty)

            theta_val = callback_value(cb_data, lagrangian.theta)

            vars_val = callback_value.(cb_data, lagrangian.vars)

            current_obj = callback_value(cb_data, primal_obj)

            obj, worst_case = solve_second_stage_RO_lagrangianUncertainty(twoROmodel, instance, solution_xN1, sol_dual_uncertainty)

            tm= [t for t in 1:T if worst_case[t]>0.1]


            # println((tm, current_obj, current_obj + obj - theta_val, obj, theta_val))

            if obj<= 0.999999*theta_val

                intercept = obj + sum(
                    worst_case[t] * sol_dual_uncertainty[t] for t in 1:T)

                cstr=@build_constraint(lagrangian.theta <= intercept - sum(
                    worst_case[t] * dual_var_uncertainty[t] for t in 1:T))
                MOI.submit(lagrangian.model, MOI.LazyConstraint(cb_data), cstr)
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

    sol_dual_uncertainty = JuMP.value.(lagrangian.dual_var_uncertainty)

    theta_val = JuMP.value(lagrangian.theta)

    dual_demand = JuMP.value.(lagrangian.dual_demand)

    return worst_case_cost_obj, worst_case
end

function get_price_demand_RO(twoROmodel::twoRO, solution_xN1::Vector{Matrix{Float64}}, solution_uncertainty::Dict{Int64, Float64}, instance::Instance)
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
    Next=instance.Next
    Buses=1:size(Next)[1]

    solution_is_onN1 = solution_xN1[1]

    for i in 1:N1
        for t in 0:T
            fix(is_on[i,t], solution_is_onN1[i,t+1]; force=true)
        end
    end

    for t in 1:T
        fix(uncertainty[t], solution_uncertainty[t]; force=true)
    end

    JuMP.optimize!(subproblem.model)

    solution_is_on = Dict((i,t) => round(JuMP.value(is_on[i,t])) for i in 1:N for t in 0:T)
    solution_start_up = Dict((i,t) => round(JuMP.value(start_up[i,t])) for i in 1:N for t in 1:T)
    solution_start_down = Dict((i,t) => round(JuMP.value(start_down[i,t])) for i in 1:N for t in 1:T)

    for i in 1:N
        for t in 0:T
            JuMP.fix(is_on[i,t], solution_is_on[i,t]; force=true)
        end
    end
    for i in 1:N
        for t in 1:T
            JuMP.fix(start_up[i,t], solution_start_up[i,t]; force=true)
            JuMP.fix(start_down[i,t], solution_start_down[i,t]; force=true)
        end
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
    for i in 1:N
        for t in 0:T
            JuMP.unfix(is_on[i,t])
        end
    end
    for i in 1:N
        for t in 1:T
            JuMP.unfix(start_up[i,t])
            JuMP.unfix(start_down[i,t])
        end
    end

    for t in 1:T
        JuMP.unfix(uncertainty[t])
    end

    return obj, bound, price_demand
end

function solve_price_relaxation(twoROmodel::twoRO, instance::Instance, price_demand::Matrix{Float64}, solution_xN1::Vector{Matrix{Float64}}, worst_case::Dict{Int64, Float64})
    """
    Get worst-case cost from Lagrangian relaxation
    """

    T= instance.TimeHorizon
    N = instance.N
    N1 = instance.N1
    N2 = instance.N - N1
    thermal_units=instance.Thermalunits
    thermal_units1 = thermal_units[1:N1]
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    Numlines=length(instance.Lines)

    relaxation1 = twoROmodel.priceRelaxation.model1
    model1 = relaxation1.model
    muup = relaxation1.muup
    mudown = relaxation1.mudown
    is_on1 = relaxation1.is_on
    start_up1 = relaxation1.start_up
    start_down1 = relaxation1.start_down
    power1 = relaxation1.power
    power_shedding1 = relaxation1.power_shedding
    power_curtailement1 = relaxation1.power_curtailement
    flow1 = relaxation1.flow
    for i in 1:N1
        for t in 0:T
            fix(is_on1[i,t], solution_xN1[1][i,t+1]; force=true)
        end
    end
    for i in 1:N1
        for t in 1:T
            fix(start_up1[i,t], solution_xN1[2][i,t]; force=true)
            fix(start_down1[i,t], solution_xN1[3][i,t]; force=true)
        end    
    end

    primal_obj = JuMP.objective_function(model1)

    new_obj_function = @expression(model1, primal_obj - sum(price_demand[t, unit.Bus] * power1[unit.name, t] for unit in thermal_units1 for t in 1:T)-sum(price_demand[t,b] * (power_shedding1[b,t] - power_curtailement1[b,t]) for b in Buses for t in 1:T)-sum(price_demand[t,b] * flow1[(line.id, t)] for line in Lines for t in 1:T for b in Buses if line.b2==b) + sum(price_demand[t,b] * flow1[(line.id, t)] for line in Lines for t in 1:T for b in Buses if line.b1==b))
    JuMP.set_objective_function(
        model1,
        new_obj_function
    )

    JuMP.optimize!(model1)

    dual_muup = JuMP.dual.(muup)
    dual_mudown = JuMP.dual.(mudown)
    
    obj1 = JuMP.objective_value(model1)
    bound1 = JuMP.objective_bound(model1)

    for i in 1:N1
        for t in 0:T
            JuMP.unfix(is_on1[i,t])
        end
    end
    for i in 1:N1
        for t in 1:T
            JuMP.unfix(start_up1[i,t])
            JuMP.unfix(start_down1[i,t])
        end
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
    bound2 = JuMP.objective_bound(model2)

    JuMP.set_objective_function(model2, primal_obj2)

    intercept = sum(price_demand[t,b] * instance.Demandbus[b][t]*(1+1.96*0.025*worst_case[t]) for b in Buses for t in 1:T)

    return ResultsPriceRelaxation(obj1+obj2+intercept, dual_muup, dual_mudown, price_demand)
end

function get_SB_cut_RO(twoROmodel::twoRO, solution_xN1::Vector{Matrix{Float64}}, solution_uncertainty::Dict{Int64, Float64}, instance::Instance)
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

    solution_is_on = solution_xN1[1]

    for i in 1:N1
        for t in 0:T
            fix(is_on[i,t], solution_xN1[1][i,t+1]; force=true)
        end
        for t in 1:T
            JuMP.fix(start_up[i,t], solution_xN1[2][i,t]; force=true)
            JuMP.fix(start_down[i,t], solution_xN1[3][i,t]; force=true)
        end
    end

    for t in 1:T
        fix(uncertainty[t], solution_uncertainty[t]; force=true)
    end

    undo_relax = JuMP.relax_integrality(subproblem.model)

    JuMP.optimize!(subproblem.model)

    sol_dual_var_is_on = Dict((i,t) => JuMP.dual(JuMP.FixRef(is_on[i,t])) for i in 1:N1 for t in 0:T)
    sol_dual_var_start_up = Dict((i,t) => JuMP.dual(JuMP.FixRef(start_up[i,t])) for i in 1:N1 for t in 1:T)
    sol_dual_var_start_down = Dict((i,t) => JuMP.dual(JuMP.FixRef(start_down[i,t])) for i in 1:N1 for t in 1:T)

    undo_relax()

    for i in 1:N1
        for t in 0:T
            JuMP.unfix(is_on[i,t])
        end
        for t in 1:T
            JuMP.unfix(start_up[i,t])
            JuMP.unfix(start_down[i,t])
        end
    end

    primal_obj = JuMP.objective_function(subproblem.model)

    JuMP.set_objective_function(
        subproblem.model,
        @expression(subproblem.model, primal_obj + sum(
            sol_dual_var_is_on[i,t] * (solution_xN1[1][i,t+1] - is_on[i,t]) for
            i in 1:N1 for t in 0:T) + sum(
            sol_dual_var_start_up[i,t] * (solution_xN1[2][i,t] - start_up[i,t]) for
            i in 1:N1 for t in 1:T) + sum(
            sol_dual_var_start_down[i,t] * (solution_xN1[3][i,t] - start_down[i,t]) for
            i in 1:N1 for t in 1:T))
    )

    JuMP.optimize!(subproblem.model)

    bound = JuMP.objective_bound(subproblem.model)

    sol_dual_var = [sol_dual_var_is_on, sol_dual_var_start_up, sol_dual_var_start_down]

    for t in 1:T
        JuMP.unfix(uncertainty[t])
    end

    JuMP.set_objective_function(subproblem.model, primal_obj)

    return bound, sol_dual_var
end