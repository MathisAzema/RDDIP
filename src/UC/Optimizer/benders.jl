function benders_callback(instance, options;silent=true, force=1, S::Int64, batch=1, gap=0.05, timelimit=10)
        
    """
    Solve the two stage SUC with Benders' Decomposition
    """

    T= instance.TimeHorizon
    N=instance.N
    thermal_units=values(instance.Thermalunits)

    master_pb=options.master_problem(instance, silent=silent, S=S)
    oracle_pb=options.oracle_problem(instance)
    
    if options._add_optimality_cuts==_add_optimality_cuts_extended_exact
        for unit in thermal_units
            for (a,b) in unit.intervals
                instance.model_Q_jab[unit.name,a,b]=initialize_model_Q_jab(T, unit, a, b, instance.optimizer)
            end
        end
    end

    infty=1e9
    LB=1
    start = time()

    Time_subproblem=[]

    LB=1
    UB=infty
    k=0

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

            second_stage_cost_ub, thermal_cost_scenario = options.add_cut(cb_data, options, master_pb, oracle_pb, instance, Time_subproblem, solution_x, solution_gamma; force=force, S=S, batch=batch, gap=gap)
            if second_stage_cost_ub<= 0.999999*UB
                UB=second_stage_cost_ub
                cstr=@build_constraint(master_pb[:thermal_cost] + master_pb[:thermal_fixed_cost] <= second_stage_cost_ub)
                MOI.submit(master_pb, MOI.LazyConstraint(cb_data), cstr)
                if k>=10 #Force to the new best solution after 10 iterations
                    vars= vcat([master_pb[:is_on][i,t] for i in 1:N for t in 0:T], [master_pb[:start_up][i,t] for i in 1:N for t in 1:T], [master_pb[:start_down][i,t] for i in 1:N for t in 1:T], [master_pb[:gamma][i, [a,b]] for i in 1:N for (a,b) in instance.Thermalunits[i].intervals],
                        [master_pb[:thermal_fixed_cost], master_pb[:thermal_cost]], [master_pb[:thermal_fuel_cost][s] for s in 1:S])
                    
                    vals= vcat([solution_is_on[i,t+1] for i in 1:N for t in 0:T], [solution_start_up[i,t] for i in 1:N for t in 1:T], [solution_start_down[i,t] for i in 1:N for t in 1:T], [solution_gamma[i,a,b] for i in 1:N for (a,b) in instance.Thermalunits[i].intervals],
                        [callback_value.(cb_data, master_pb[:thermal_fixed_cost]), sum(thermal_cost_scenario[s] for s in 1:S)/S], [thermal_cost_scenario[s] for s in 1:S])
                    
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

        solution_gamma = Dict{Tuple{Int, Int, Int}, Float64}()
        gamma_val=value.(master_pb[:gamma])
        for unit in thermal_units
            for (a,b) in unit.intervals
                solution_gamma[unit.name,a,b]=gamma_val[unit.name, [a,b]]
            end
        end
        solution_x = [solution_is_on, solution_start_up, solution_start_down]

        current_solution = [solution_x, solution_gamma]

        return instance.name, computation_time, objective_value(master_pb), objective_bound(master_pb), current_solution, S,batch, Time_subproblem, gap, force
    else
        return nothing
    end
end

function benders3bin_callback(instance, options;silent=true, force=1, S::Int64, batch=1, gap=0.05, timelimit=10)
    """
    Solve the two stage SUC with Benders' Decomposition in the 3bin formulation
    """
    T= instance.TimeHorizon
    N=instance.N

    master_pb=options.master_problem(instance, silent=silent, S=S)
    oracle_pb=options.oracle_problem(instance)

    infty=1e9
    LB=1
    start = time()

    Time_subproblem=[]

    LB=1
    UB=infty
    k=0

    function my_callback_function(cb_data, cb_where::Cint)

        if cb_where==GRB_CB_MIPSOL
            k+=1
    
            Gurobi.load_callback_variable_primal(cb_data, cb_where)

            solution_is_on = convert(Matrix{Float64}, round.(callback_value.(cb_data, master_pb[:is_on])))
            solution_start_up = convert(Matrix{Float64}, round.(callback_value.(cb_data, master_pb[:start_up])))
            solution_start_down = convert(Matrix{Float64}, round.(callback_value.(cb_data, master_pb[:start_down])))

            solution_x = [solution_is_on, solution_start_up, solution_start_down]

            second_stage_cost_ub, thermal_cost_scenario = options.add_cut(cb_data, options, master_pb, oracle_pb, instance, Time_subproblem, solution_x; force=force, S=S, batch=batch, gap=gap)
            if second_stage_cost_ub<= 0.999999*UB
                UB=second_stage_cost_ub
                cstr=@build_constraint(master_pb[:thermal_cost] + master_pb[:thermal_fixed_cost] <= second_stage_cost_ub)
                MOI.submit(master_pb, MOI.LazyConstraint(cb_data), cstr)
                if k>=10 #Force to the new best solution after 10 iterations
                    vars= vcat([master_pb[:is_on][i,t] for i in 1:N for t in 0:T], [master_pb[:start_up][i,t] for i in 1:N for t in 1:T], [master_pb[:start_down][i,t] for i in 1:N for t in 1:T],
                        [master_pb[:thermal_fixed_cost], master_pb[:thermal_cost]], [master_pb[:thermal_fuel_cost][s] for s in 1:S])
                    
                    vals= vcat([solution_is_on[i,t+1] for i in 1:N for t in 0:T], [solution_start_up[i,t] for i in 1:N for t in 1:T], [solution_start_down[i,t] for i in 1:N for t in 1:T],
                        [callback_value.(cb_data, master_pb[:thermal_fixed_cost]), sum(thermal_cost_scenario[s] for s in 1:S)/S], [thermal_cost_scenario[s] for s in 1:S])
                    
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

        return instance.name, computation_time, objective_value(master_pb), objective_bound(master_pb), current_solution, S,batch, Time_subproblem, gap, force
    else
        return nothing
    end
end