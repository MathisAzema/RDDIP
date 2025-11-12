function subproblem_builder_UC(instance::RDDIP.Instance, force::Float64, subproblem::Model, node::Int)
    # State variables
    K = 8
    N=instance.N
    T= instance.TimeHorizon
    thermal_units=values(instance.Thermalunits)
    Next=instance.Next
    Buses=1:size(Next)[1]
    Lines=values(instance.Lines)
    BusWind=instance.BusWind
    NumWindfarms=length(BusWind)

    constraints_uncertainty = JuMP.ConstraintRef[]

    
    @variable(subproblem, is_on[i = 1:N], Bin, RDDIP.State, initial_value = thermal_units[i].InitUpDownTime>0) #initial value fictive
    @variable(subproblem, start_up[i = 1:N], Bin)
    @variable(subproblem, start_down[i = 1:N], Bin)

    idx_list = [0 for unit in thermal_units]

    for unit in thermal_units
        initpower = unit.InitialPower
        power_levels = [0.0; [unit.MinPower + (k-1)*(unit.MaxPower - unit.MinPower)/(K-1) for k in 1:K]]
        idx = argmin(abs.(power_levels .- initpower))
        idx_list[unit.name] = idx-1
    end

    init_integer = [[idx_list[i]==k for i in 1:N] for k in 1:K]


    @variable(subproblem, power_integer[i = 1:N, k in 1:K], Bin, RDDIP.State, initial_value = init_integer[k][i])
    @variable(subproblem, power_dev[i = 1:N])
    @variable(subproblem, power[i = 1:N]>=0)
    @variable(subproblem, power_real[i = 1:N]>=0)
    @variable(subproblem, power_prev[i = 1:N]>=0)

    @constraint(subproblem, borne_sup_dev[i = 1:N], power_dev[i] <= 0.1*is_on[i].out*(thermal_units[i].MaxPower - thermal_units[i].MinPower)/(K-1))
    @constraint(subproblem,  borne_inf_dev[i = 1:N], power_dev[i] >= -0.1*is_on[i].out*(thermal_units[i].MaxPower - thermal_units[i].MinPower)/(K-1))
    @constraint(subproblem,  def_real[i = 1:N], sum(power_integer[i,k].out*(thermal_units[i].MinPower + (k-1)*(thermal_units[i].MaxPower - thermal_units[i].MinPower)/(K-1)) for k in 1:K) + power_dev[i] == power_real[i])
    @constraint(subproblem,  def_power[i = 1:N], sum(power_integer[i,k].out*(thermal_units[i].MinPower + (k-1)*(thermal_units[i].MaxPower - thermal_units[i].MinPower)/(K-1)) for k in 1:K) == power[i])
    @constraint(subproblem,  def_prev[i = 1:N], sum(power_integer[i,k].in*(thermal_units[i].MinPower + (k-1)*(thermal_units[i].MaxPower - thermal_units[i].MinPower)/(K-1)) for k in 1:K) == power_prev[i])
    @constraint(subproblem,  limit_out[i = 1:N], sum(power_integer[i,k].out for k in 1:K) == is_on[i].out)
    @constraint(subproblem,  limit_in[i = 1:N], sum(power_integer[i,k].in for k in 1:K) == is_on[i].in)

    # @constraint(subproblem, [i in 1:N], is_on[i].out == 1)
    # if node == 3
    #     @constraint(subproblem, [i = 1:4], power_integer[i,3].out == 1)
    #     @constraint(subproblem, power_integer[5,1].out == 1)
    # end

    initup=[[0 for k in 0:thermal_units[i].MinUpTime-1] for i in 1:N]
    initdown=[[0 for k in 0:thermal_units[i].MinDownTime-1] for i in 1:N]
    for i in 1:N
        if thermal_units[i].InitUpDownTime!=nothing
            if thermal_units[i].InitUpDownTime>0
                for k in 0:thermal_units[i].MinUpTime-1
                    initup[i][k+1] = (k==min(thermal_units[i].InitUpDownTime-1, thermal_units[i].MinUpTime-1))
                end
            elseif thermal_units[i].InitUpDownTime<0
                for k in 0:thermal_units[i].MinDownTime-1
                    initdown[i][k+1] = (k==min(-thermal_units[i].InitUpDownTime-1, thermal_units[i].MinDownTime-1))
                end
            end
        end
    end
    # @variable(subproblem, u[i = 1:N, k in 0:thermal_units[i].MinUpTime-1], Bin, RDDIP.State, initial_value = initup[i][k+1])
    # @constraint(subproblem, [i = 1:N], u[i,0].out == start_up[i])
    # @constraint(subproblem, [i = 1:N, k in 1:thermal_units[i].MinUpTime-1], u[i,k].out == u[i,k-1].in)

    # @variable(subproblem, v[i = 1:N, k in 0:thermal_units[i].MinDownTime-1], Bin, RDDIP.State, initial_value = initdown[i][k+1])
    # @constraint(subproblem, [i = 1:N], v[i,0].out == start_down[i])
    # @constraint(subproblem, [i = 1:N, k in 1:thermal_units[i].MinDownTime-1], v[i,k].out == v[i,k-1].in)

    # @constraint(subproblem, [i = 1:N], sum(u[i,k].out for k in 0:thermal_units[i].MinUpTime-1) <= is_on[i].out)
    # @constraint(subproblem, [i = 1:N], sum(v[i,k].out for k in 0:thermal_units[i].MinDownTime-1) <= 1 - is_on[i].out)

    @constraint(subproblem,  def_is_on[i = 1:N], is_on[i].out == is_on[i].in + start_up[i] - start_down[i])
    @constraint(subproblem,  limit_start_up[i = 1:N], start_up[i] <= is_on[i].out)
    @constraint(subproblem,  limit_start_down[i = 1:N], start_down[i] <= 1 - is_on[i].out)

    Numlines=length(instance.Lines)
        # Control variables
    @variables(subproblem, begin
        power_shedding[b in Buses] >= 0
        power_curtailement[b in Buses] >= 0
        θ[b in Buses] 
        flow[l in 1:Numlines]
    end)

    t=node

        # Random variables
    @variable(subproblem, error_forecast[k in 1:NumWindfarms])
    M=NumWindfarms
    Ω = [[0.0 for k in 1:NumWindfarms] for s in 1:3*M]
    P = [1/(3*M) for s in 1:3*M]
    for s in 1:M
        Ω[s][s] = -1.0
        Ω[s+M][s] = 0.0
        Ω[s+2M][s] = 1.0
    end
    # M = 3
    # Ω = [[(-1.0+(s-1)*1.0) for b in BusWind] for s in 1:M]
    # P = [1/M for s in 1:M]
    # Ω = [[0.0 for b in BusWind]]
    # P = [1.0]
    if t == 1
        Ω = [[0.0 for b in BusWind]]
        P = [1.0]
    end
    @variable(subproblem, uncertainty[s in 1:length(P)], Bin, RDDIP.Uncertain)
    cstr_uncertainty1 = [@constraint(subproblem, sum_uncertainty, sum(uncertainty[s].var for s in 1:length(P)) == 1)]
    for c in cstr_uncertainty1
        push!(constraints_uncertainty, c)
    end
    @constraint(subproblem, def_error_forecast[k in 1:NumWindfarms], error_forecast[k] == sum(Ω[s][k]*uncertainty[s].var for s in 1:length(P)))

    Scenario = [Dict{Symbol, Float64}() for s in 1:length(P)]
    # println(Scenario)
    for s in 1:length(P)
        for k in 1:length(P)
            # println((s,k, 1.0*(s==k), Symbol(JuMP.name(uncertainty[k].var))))
            Scenario[s][Symbol(JuMP.name(uncertainty[k].var))] = 1.0*(s==k)
        end
    end


        # Transition function and constraints
    @constraints(
        subproblem,
        begin
            def_power_limit_upper[i = 1: N], power[i] <= thermal_units[i].MaxPower * is_on[i].out
            def_power_limit_lower[i = 1: N], power[i] >= thermal_units[i].MinPower * is_on[i].out
            def_power_real_limit_upper[i = 1: N], power_real[i] <= thermal_units[i].MaxPower * is_on[i].out
            def_power_real_limit_lower[i = 1: N], power_real[i] >= thermal_units[i].MinPower * is_on[i].out
            def_ramp_up[i = 1:N], power_real[i] - power_prev[i] <= (-thermal_units[i].DeltaRampUp)*start_up[i] + (thermal_units[i].MinPower + thermal_units[i].DeltaRampUp)*is_on[i].out - (thermal_units[i].MinPower)*is_on[i].in
            def_ramp_down[i = 1: N], power_prev[i] - power_real[i] <= (-thermal_units[i].DeltaRampDown)*start_down[i] + (thermal_units[i].MinPower + thermal_units[i].DeltaRampDown)*is_on[i].in - (thermal_units[i].MinPower)*is_on[i].out
            def_flow_limit_upper[line in Lines], flow[line.id] <= line.Fmax
            def_flow_limit_lower[line in Lines], flow[line.id] >= -line.Fmax
            def_flow[line in Lines], flow[line.id] == line.B12*(θ[line.b1]-θ[line.b2])
        end
    )
    @constraint(subproblem, demand[b in Buses], sum(power_real[unit.name] for unit in thermal_units if unit.Bus==b) - power_curtailement[b] + power_shedding[b] + sum(flow[line.id] for line in Lines if line.b2==b) - sum(flow[line.id] for line in Lines if line.b1==b) == instance.Demandbus[b][t]*(1+force*1.96*0.025*sum(error_forecast[w] for w in 1:NumWindfarms if BusWind[w]==b)))
        # Stage-objective
    @stageobjective(subproblem, sum(unit.LinearTerm*power_real[unit.name] for unit in thermal_units)+sum(RDDIP.SHEDDING_COST*power_shedding[b]+RDDIP.CURTAILEMENT_COST*power_curtailement[b] for b in Buses) + sum(unit.ConstTerm*is_on[unit.name].out+unit.StartUpCost*start_up[unit.name]+unit.StartDownCost*start_down[unit.name] for unit in thermal_units))

    return Scenario, P, constraints_uncertainty

end