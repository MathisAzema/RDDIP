function benders_RO_callback(instance;silent=true, Γ=0, force=1, gap=0.05, timelimit=10)
        
    """
    Solve the two stage SUC with Benders' Decomposition
    """

    T= instance.TimeHorizon
    N=instance.N
    thermal_units=values(instance.Thermalunits)

    master_pb=master_RO_problem_extended(instance, silent=silent)
    oracle_pb = oracle_RO_problem(instance; silent=true, Γ=Γ)
    
    infty=1e9
    LB=1
    start = time()

    LB=1
    UB=infty
    k=0
    Time_subproblem = []

    function my_callback_function(cb_data, cb_where::Cint)

        if cb_where==GRB_CB_MIPSOL
            k+=1
    
            Gurobi.load_callback_variable_primal(cb_data, cb_where)

            solution_is_on = convert(Matrix{Float64}, round.(callback_value.(cb_data, master_pb[:is_on])))
            solution_start_up = convert(Matrix{Float64}, round.(callback_value.(cb_data, master_pb[:start_up])))
            solution_start_down = convert(Matrix{Float64}, round.(callback_value.(cb_data, master_pb[:start_down])))

            solution_gamma = Dict{Tuple{Int, Int, Int}, Float64}()
            gamma_val=round.(callback_value.(cb_data, master_pb[:gamma]))
            for unit in thermal_units
                for (a,b) in unit.intervals
                    solution_gamma[unit.name,a,b]=gamma_val[unit.name, [a,b]]
                end
            end
            solution_x = [solution_is_on, solution_start_up, solution_start_down]

            second_stage_cost_ub, worst_case_cost = add_cut_RO(cb_data, master_pb, oracle_pb, instance, Time_subproblem, solution_x, solution_gamma; gap=gap)

            if second_stage_cost_ub<= 0.999999*UB
                UB=second_stage_cost_ub
                cstr=@build_constraint(master_pb[:thermal_cost] + master_pb[:thermal_fixed_cost] <= second_stage_cost_ub)
                MOI.submit(master_pb, MOI.LazyConstraint(cb_data), cstr)
                if k>=10 #Force to the new best solution after 10 iterations
                    vars= vcat([master_pb[:is_on][i,t] for i in 1:N for t in 0:T], [master_pb[:start_up][i,t] for i in 1:N for t in 1:T], [master_pb[:start_down][i,t] for i in 1:N for t in 1:T], [master_pb[:gamma][i, [a,b]] for i in 1:N for (a,b) in instance.Thermalunits[i].intervals],
                        [master_pb[:thermal_fixed_cost], master_pb[:thermal_cost]])
                    
                    vals= vcat([solution_is_on[i,t+1] for i in 1:N for t in 0:T], [solution_start_up[i,t] for i in 1:N for t in 1:T], [solution_start_down[i,t] for i in 1:N for t in 1:T], [solution_gamma[i,a,b] for i in 1:N for (a,b) in instance.Thermalunits[i].intervals],
                        [callback_value.(cb_data, master_pb[:thermal_fixed_cost]), worst_case_cost])
                    
                    MOI.submit(master_pb, MOI.HeuristicSolution(cb_data), vars, vals)
                end
            end
        end
        return
    end

    set_optimizer_attribute(master_pb, "TimeLimit", timelimit-(time() - start))
    set_optimizer_attribute(master_pb, "LazyConstraints", 1)
    MOI.set(master_pb, Gurobi.CallbackFunction(), my_callback_function)
    start = time()
    newgap=gap/100
    set_optimizer_attribute(master_pb, "MIPGap", newgap)
    optimize!(master_pb)
    computation_time = time() - start

    feasibleSolutionFound = primal_status(master_pb) == MOI.FEASIBLE_POINT

    if feasibleSolutionFound

        solution_is_on = convert(Matrix{Float64}, value.(master_pb[:is_on]))
        solution_start_up = convert(Matrix{Float64}, value.(master_pb[:start_up]))
        solution_start_down = convert(Matrix{Float64}, value.(master_pb[:start_down]))

        solution_x = [solution_is_on, solution_start_up, solution_start_down]

        current_solution = solution_x

        println((value(master_pb[:thermal_fixed_cost]), value(master_pb[:thermal_cost])))

        return instance.name, computation_time, objective_value(master_pb), objective_bound(master_pb), current_solution, Time_subproblem, gap, force
    else
        return Time_subproblem
    end
end

function benders_RO_bin_callback(instance;silent=true, Γ=0, force=1, gap=0.05, timelimit=10)
        
    """
    Solve the two stage SUC with Benders' Decomposition
    """

    T= instance.TimeHorizon
    N=instance.N
    N1=instance.N1
    thermal_units=values(instance.Thermalunits)
    thermal_units_N1=thermal_units[1:N1]

    master_pb=master_RO_bin_problem_extended(instance, silent=silent)
    lagrangian = Lagrangian_RO_problem(instance; silent=true, Γ=Γ)
    lagrangianUncertainty = Lagrangian_uncertainty_RO_problem(instance; silent=true, Γ=Γ)
    subproblem = subproblemRO(instance, Γ)
    oracleContinuousRO = oracle_Continuous_RO_problem(instance; silent=true, Γ=Γ)
    pricerelaxation = initialize_price_relaxation(instance)

    twoROmodel = twoRO(master_pb, lagrangian, lagrangianUncertainty, subproblem, oracleContinuousRO, pricerelaxation)

    # return twoROmodel
    
    infty=1e9
    LB=1
    start = time()

    LB=1
    UB=infty
    k=0
    Time_subproblem = []

    master_pb=twoROmodel.master_pb

    function my_callback_function(cb_data, cb_where::Cint)

        if cb_where==GRB_CB_MIPSOL
            k+=1
            # println(k)
    
            Gurobi.load_callback_variable_primal(cb_data, cb_where)

            solution_is_on = convert(Matrix{Float64}, round.(callback_value.(cb_data, master_pb[:is_on])))
            solution_start_up = convert(Matrix{Float64}, round.(callback_value.(cb_data, master_pb[:start_up])))
            solution_start_down = convert(Matrix{Float64}, round.(callback_value.(cb_data, master_pb[:start_down])))

            solution_gamma = Dict{Tuple{Int, Int, Int}, Float64}()
            gamma_val=round.(callback_value.(cb_data, master_pb[:gamma]))
            for unit in thermal_units_N1
                for (a,b) in unit.intervals
                    solution_gamma[unit.name,a,b]=gamma_val[unit.name, [a,b]]
                end
            end
            solution_x = [solution_is_on, solution_start_up, solution_start_down]

            update_UB, second_stage_cost_ub, worst_case_cost = add_cut_RO_bin(cb_data, twoROmodel, instance, Time_subproblem, solution_x, solution_gamma; gap=gap)

            # println(("H",k, update_UB, second_stage_cost_ub, UB))

            if second_stage_cost_ub<= 0.999999*UB && update_UB && k>=10
                UB=second_stage_cost_ub
                # println("UB : ", UB)
                cstr=@build_constraint(master_pb[:thermal_cost] + master_pb[:thermal_fixed_cost] <= second_stage_cost_ub)
                MOI.submit(master_pb, MOI.LazyConstraint(cb_data), cstr)
                if k>=10 #Force to the new best solution after 10 iterations
                    vars= vcat([master_pb[:is_on][i,t] for i in 1:N1 for t in 0:T], [master_pb[:start_up][i,t] for i in 1:N1 for t in 1:T], [master_pb[:start_down][i,t] for i in 1:N1 for t in 1:T], [master_pb[:gamma][i, [a,b]] for i in 1:N1 for (a,b) in instance.Thermalunits[i].intervals],
                        [master_pb[:thermal_fixed_cost], master_pb[:thermal_cost]])
                    
                    vals= vcat([solution_is_on[i,t+1] for i in 1:N1 for t in 0:T], [solution_start_up[i,t] for i in 1:N1 for t in 1:T], [solution_start_down[i,t] for i in 1:N1 for t in 1:T], [solution_gamma[i,a,b] for i in 1:N1 for (a,b) in instance.Thermalunits[i].intervals],
                        [callback_value.(cb_data, master_pb[:thermal_fixed_cost]), worst_case_cost])
                    
                    MOI.submit(master_pb, MOI.HeuristicSolution(cb_data), vars, vals)
                end
            end
        end
        return
    end

    set_optimizer_attribute(master_pb, "TimeLimit", timelimit-(time() - start))
    set_optimizer_attribute(master_pb, "LazyConstraints", 1)
    MOI.set(master_pb, Gurobi.CallbackFunction(), my_callback_function)
    start = time()
    newgap=gap/100
    set_optimizer_attribute(master_pb, "MIPGap", newgap)
    optimize!(master_pb)
    computation_time = time() - start

    feasibleSolutionFound = primal_status(master_pb) == MOI.FEASIBLE_POINT

    if feasibleSolutionFound

        solution_is_on = convert(Matrix{Float64}, value.(master_pb[:is_on]))
        solution_start_up = convert(Matrix{Float64}, value.(master_pb[:start_up]))
        solution_start_down = convert(Matrix{Float64}, value.(master_pb[:start_down]))

        solution_x = [solution_is_on, solution_start_up, solution_start_down]

        current_solution = solution_x

        return instance.name, computation_time, objective_value(master_pb), objective_bound(master_pb), current_solution, Time_subproblem, gap, force, Γ, k
    else
        return nothing
    end
end

function benders_RO_bin_callback2(instance;silent=true, Γ=0, force=1, gap=0.05, timelimit=10)
        
    """
    Solve the two stage SUC with Benders' Decomposition
    """

    T= instance.TimeHorizon
    N=instance.N
    N1=instance.N1
    thermal_units=values(instance.Thermalunits)
    thermal_units_N1=thermal_units[1:N1]

    master_pb=master_RO_bin_problem_extended(instance, silent=silent)
    lagrangian = Lagrangian_RO_problem(instance; silent=true, Γ=Γ)
    lagrangianUncertainty = Lagrangian_uncertainty_RO_problem(instance; silent=true, Γ=Γ)
    subproblem = subproblemRO(instance, Γ)
    oracleContinuousRO = oracle_Continuous_RO_problem(instance; silent=true, Γ=Γ)
    pricerelaxation = initialize_price_relaxation(instance)

    twoROmodel = twoRO(master_pb, lagrangian, lagrangianUncertainty, subproblem, oracleContinuousRO, pricerelaxation)

    # return twoROmodel
    
    infty=1e9
    LB=1
    start = time()

    LB=1
    UB=infty
    k=0
    Time_subproblem = []

    master_pb=twoROmodel.master_pb

    function my_callback_function(cb_data, cb_where::Cint)

        if cb_where==GRB_CB_MIPSOL
            k+=1
            # println(k)
    
            Gurobi.load_callback_variable_primal(cb_data, cb_where)

            solution_is_on = convert(Matrix{Float64}, round.(callback_value.(cb_data, master_pb[:is_on])))
            solution_start_up = convert(Matrix{Float64}, round.(callback_value.(cb_data, master_pb[:start_up])))
            solution_start_down = convert(Matrix{Float64}, round.(callback_value.(cb_data, master_pb[:start_down])))

            solution_gamma = Dict{Tuple{Int, Int, Int}, Float64}()
            gamma_val=round.(callback_value.(cb_data, master_pb[:gamma]))
            for unit in thermal_units_N1
                for (a,b) in unit.intervals
                    solution_gamma[unit.name,a,b]=gamma_val[unit.name, [a,b]]
                end
            end
            solution_x = [solution_is_on, solution_start_up, solution_start_down]

            update_UB, second_stage_cost_ub, worst_case_cost = add_cut_RO_bin2(cb_data, twoROmodel, instance, Time_subproblem, solution_x, solution_gamma, Γ; gap=gap)

            # println(("H",k, update_UB, second_stage_cost_ub, UB))

            if second_stage_cost_ub<= 0.999999*UB && update_UB && k>=10
                UB=second_stage_cost_ub
                # println("UB : ", UB)
                cstr=@build_constraint(master_pb[:thermal_cost] + master_pb[:thermal_fixed_cost] <= second_stage_cost_ub)
                MOI.submit(master_pb, MOI.LazyConstraint(cb_data), cstr)
                if k>=10 #Force to the new best solution after 10 iterations
                    vars= vcat([master_pb[:is_on][i,t] for i in 1:N1 for t in 0:T], [master_pb[:start_up][i,t] for i in 1:N1 for t in 1:T], [master_pb[:start_down][i,t] for i in 1:N1 for t in 1:T], [master_pb[:gamma][i, [a,b]] for i in 1:N1 for (a,b) in instance.Thermalunits[i].intervals],
                        [master_pb[:thermal_fixed_cost], master_pb[:thermal_cost]])
                    
                    vals= vcat([solution_is_on[i,t+1] for i in 1:N1 for t in 0:T], [solution_start_up[i,t] for i in 1:N1 for t in 1:T], [solution_start_down[i,t] for i in 1:N1 for t in 1:T], [solution_gamma[i,a,b] for i in 1:N1 for (a,b) in instance.Thermalunits[i].intervals],
                        [callback_value.(cb_data, master_pb[:thermal_fixed_cost]), worst_case_cost])
                    
                    MOI.submit(master_pb, MOI.HeuristicSolution(cb_data), vars, vals)
                end
            end
        end
        return
    end

    set_optimizer_attribute(master_pb, "TimeLimit", timelimit-(time() - start))
    set_optimizer_attribute(master_pb, "LazyConstraints", 1)
    MOI.set(master_pb, Gurobi.CallbackFunction(), my_callback_function)
    start = time()
    newgap=gap/100
    set_optimizer_attribute(master_pb, "MIPGap", newgap)
    optimize!(master_pb)
    computation_time = time() - start

    feasibleSolutionFound = primal_status(master_pb) == MOI.FEASIBLE_POINT

    if feasibleSolutionFound

        solution_is_on = convert(Matrix{Float64}, value.(master_pb[:is_on]))
        solution_start_up = convert(Matrix{Float64}, value.(master_pb[:start_up]))
        solution_start_down = convert(Matrix{Float64}, value.(master_pb[:start_down]))

        solution_x = [solution_is_on, solution_start_up, solution_start_down]

        current_solution = solution_x

        return instance.name, computation_time, objective_value(master_pb), objective_bound(master_pb), current_solution, Time_subproblem, gap, force, Γ, k
    else
        return nothing
    end
end