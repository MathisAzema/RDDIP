function add_cut_AVAR(cb_data, options::UCOptions, master_pb::JuMP.Model, oracle_pb, instance::Instance, Time_subproblem, solution_x::Vector{Matrix{Float64}}, solution_gamma::Dict{Tuple{Int64, Int64, Int64}, Float64}; force::Float64, S::Int64, batch=1, gap, α)
    T= instance.TimeHorizon
    thermal_units=instance.Thermalunits

    thermal_cost=master_pb[:thermal_cost]
    thermal_fixed_cost=master_pb[:thermal_fixed_cost]

    thermal_cost_val=callback_value(cb_data, thermal_cost)
    thermal_fixed_cost_val=callback_value(cb_data, thermal_fixed_cost)

    z_AVAR=master_pb[:z_AVAR]
    z_AVAR_val=callback_value(cb_data, z_AVAR)

    Time_iter=[0.0 for s in 1:S]

    Pmin1, Pmax1, δ_up1, δ_down1=get_limit_power_solution(instance, solution_x)

    @objective(oracle_pb, Max, sum(oracle_pb[:λ][unit.name]*unit.InitialPower for unit in thermal_units if unit.InitialPower!=nothing)+ sum(oracle_pb[:μₘᵢₙ][unit.name,t]*Pmin1[unit.name,t] - oracle_pb[:μₘₐₓ][unit.name,t]*Pmax1[unit.name,t]+oracle_pb[:μꜛ][unit.name,t]*δ_up1[unit.name,t] +oracle_pb[:μꜜ][unit.name,t]*δ_down1[unit.name,t] for unit in thermal_units for t in 1:T)+oracle_pb[:network_cost])

    cut_parameters=[options.second_stage(instance, oracle_pb; batch=batch, scenario=s, force=force) for s in 1:S]
    for s in 1:S
        Time_iter[s]=cut_parameters[s].computation_time
    end
    push!(Time_subproblem, Time_iter)

    current_intervals=Dict{Int, Vector{Tuple{Int, Int}}}()
    for unit in thermal_units
        current_intervals[unit.name] = Vector{Tuple{Int, Int}}[]
        for (a,b) in unit.intervals
            if solution_gamma[unit.name, a,b]>=0.9 
                push!(current_intervals[unit.name], (a,b))
            end
        end
    end

    feasible_cuts = options._add_feasibility_cuts(cb_data, master_pb, instance, cut_parameters, solution_gamma, current_intervals; S=S)

    if feasible_cuts
        return Inf, [0.0 for s in 1:S]
    end

    if thermal_fixed_cost_val+z_AVAR_val+(sum([max(0,cut_parameters[s].objective_value-z_AVAR_val) for s in 1:S])/S)/(1-α)>= (1+0.01*gap/100)*(thermal_fixed_cost_val+z_AVAR_val+thermal_cost_val/(1-α))
        options._add_optimality_cuts(cb_data, master_pb, instance, cut_parameters, solution_gamma, current_intervals; S=S)
    end

    return thermal_fixed_cost_val+z_AVAR_val+(sum([max(0,cut_parameters[s].objective_value-z_AVAR_val) for s in 1:S])/S)/(1-α), [cut_parameters[s].objective_value for s in 1:S]
end

function _add_optimality_cuts_extended_AVAR(cb_data, master_pb, instance::Instance, cut_parameters::Vector{oracleResults}, solution_gamma::Dict{Tuple{Int64, Int64, Int64}, Float64}, current_intervals::Dict{Int, Vector{Tuple{Int, Int}}}; S::Int64)
    T= instance.TimeHorizon
    gamma=master_pb[:gamma]
    thermal_cost=master_pb[:thermal_cost]
    gamma_val=solution_gamma
    z_AVAR=master_pb[:z_AVAR]
    z_AVAR_val=callback_value(cb_data, z_AVAR)
    thermal_units=instance.Thermalunits
    S_pos=zeros(Int, S)
    for s in 1:S
        S_pos[s]=cut_parameters[s].objective_value-z_AVAR_val >=0 ? 1 : 0
    end
    Costab=Dict{Tuple{Int, Int, Int}, Float64}()
    for unit in thermal_units
        for (a,b) in unit.intervals
            Costab[unit.name,a,b]=0
        end
    end
    Vab=zeros(T)
    Wab=zeros(T)
    price_unit=zeros(T)

    for s in 1:S
        if S_pos[s]>=0.5
            muup=cut_parameters[s].muup
            mudown=cut_parameters[s].mudown
            price=cut_parameters[s].ν
            status = cut_parameters[s].status
            for unit in thermal_units
                price_unit=price[unit.Bus,:]
                for (a,b) in unit.intervals
                    Costab[unit.name,a,b]+= heuristic_cost(unit, a, b, price_unit, gamma_val, muup, mudown, current_intervals[unit.name], status, Vab, Wab)
                end
            end
        end
    end

    cstr=@build_constraint(sum(gamma[unit.name,[a,b]]*Costab[unit.name,a,b] for unit in thermal_units for (a,b) in unit.intervals)/S+sum(S_pos[s]*(cut_parameters[s].interceptE-z_AVAR) for s in 1:S)/S<=thermal_cost)
    MOI.submit(master_pb, MOI.LazyConstraint(cb_data), cstr)

end