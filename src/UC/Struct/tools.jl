function read_data_forecast(file)
    return CSV.read(file, DataFrame; delim=",", header=0)
end

function get_limit_power_solution(instance, solution)
    is_on=solution[1]
    start_up=solution[2]
    start_down=solution[3]
    
    T= instance.TimeHorizon
    N=instance.N
    thermal_units=instance.Thermalunits
    Pmin=Dict{Tuple{Int, Int}, Float64}()
    Pmax=Dict{Tuple{Int, Int}, Float64}()
    δ_up=Dict{Tuple{Int, Int}, Float64}()
    δ_down=Dict{Tuple{Int, Int}, Float64}()
    for unit in thermal_units
        if unit.InitialPower!=nothing
            Pmin[unit.name,0]=unit.MinPower*is_on[unit.name,0+1]
            Pmax[unit.name,0]=unit.MaxPower*is_on[unit.name,0+1]
        end
        for t in 1:T
            Pmin[unit.name,t]=unit.MinPower*is_on[unit.name,t+1]
            Pmax[unit.name,t]=unit.MaxPower*is_on[unit.name,t+1]
            δ_up[unit.name,t]=(unit.DeltaRampUp)*start_up[unit.name,t]-(unit.MinPower+unit.DeltaRampUp)*is_on[unit.name,t+1]+(unit.MinPower)*is_on[unit.name,t-1+1]
            δ_down[unit.name, t]=(unit.DeltaRampDown)*start_down[unit.name, t]-(unit.MinPower+unit.DeltaRampDown)*is_on[unit.name,t-1+1]+(unit.MinPower)*is_on[unit.name,t+1]
        end
    end
    return Pmin, Pmax, δ_up, δ_down
end

function initialize_model_Q_jab(T, unit, a, b, optimizer)
    """
    initialize model (without the objective function)
    """
    model = Model(optimizer)
    set_silent(model)
    
    @variable(model, power[t in 0:T])
    @constraint(model, init, power[0]==unit.InitialPower*(a<=0))
    @constraint(model, mumin[t in 0:T], power[t]>=unit.MinPower*(a<=t<=b-1))
    @constraint(model, mumax[t in 0:T], power[t]<=unit.MaxPower*(a<=t<=b-1))

    @constraint(model,  muup[t in max(1,a):min(T,b)], power[t]-power[t-1]<=unit.DeltaRampUp*(t>=a+1)+unit.MinPower*(t==a))
    @constraint(model,  mudown[t in max(1,a):min(T,b)], power[t-1]-power[t]<=unit.DeltaRampDown*(t<=b-1)+unit.MinPower*(t==b))

    @objective(model, Min, 0)

    return model
end

function compute_Q_jab(price, unit, model, status)
    set_objective_coefficient.(model,model[:power][1:24], status*unit.LinearTerm .- price[1:24])
    optimize!(model)

    return objective_value(model)
end

function compute_all_Q_jab(price, instance, model_Q_jab)
    thermal_units=instance.Thermalunits
    h=0
    for unit in thermal_units
        for (a,b) in unit.intervals
            compute_Q_jab(price, unit, model_Q_jab[unit.name,a,b], status)
            h+=1
        end
    end
    return h
end

function compute_phi_UP(unit, a,b)
    """
    Compute the maximum number of consecutive time steps the ramping up constraint can be active
    """
    if a<=0
        Tup=Int64(floor((unit.MaxPower-unit.InitialPower)/unit.DeltaRampUp))
        tstar=((b-1)*unit.DeltaRampDown+unit.MinPower-unit.InitialPower)/(unit.DeltaRampDown+unit.DeltaRampUp)
        return min(Tup, Int64(floor(tstar)))
    else
        tstar=((b-1)*unit.DeltaRampDown+a*unit.DeltaRampUp)/(unit.DeltaRampDown+unit.DeltaRampUp)
        return min(unit.Tup, Int64(floor(tstar))-a+1)
    end
end

function compute_phi_DOWN(unit, a,b)
    """
    Compute the maximum number of consecutive time steps the ramping down constraint can be active
    """
    if a<=0
        Tdown=Int64(1+floor((unit.MaxPower-unit.MinPower)/unit.DeltaRampDown))
        tstar=((b-1)*unit.DeltaRampDown+unit.MinPower-unit.InitialPower)/(unit.DeltaRampDown+unit.DeltaRampUp)
        return min(Tdown, b-1-Int64(floor(tstar)))
    else
        tstar=((b-1)*unit.DeltaRampDown+a*unit.DeltaRampUp)/(unit.DeltaRampDown+unit.DeltaRampUp)
        return min(unit.Tdown,b-1-Int64(floor(tstar)))
    end
end

function compute_heuristic_dual_variable_ramping_up(Vab, price, unit, a,b, phi_a_b, gamma, muup, mudown, current_intervals, status)
    """
    Compute the heuristic dual variable of the ramping up constraint in the matrix Vab
    """
    if a<=0
        for t in max(1,a):min(b,24)
            Vab[t-max(1,a)+1]=max(0.0,sum(price[k]-status*unit.LinearTerm for k in t:phi_a_b; init=0))
        end
    else
        for t in max(1,a):min(b,24)
            Vab[t-max(1,a)+1]=max(0.0,sum(price[k]-status*unit.LinearTerm for k in t:a+phi_a_b-1; init=0))
        end
    end
    if min(b,24)-max(1,a)>= unit.Tup+unit.Tdown
        for (c,d) in current_intervals
            tmin=max(1,a,c)+unit.Tup*(1-(c==a))
            tmax=min(24,b,d)-unit.Tdown*(1-(b==d))
            for t in tmin:tmax
                Vab[t-max(1,a)+1]=muup[unit.name,t]
            end
        end
    elseif abs(1-gamma[unit.name,a,b])<=0.1
        for t in max(1,a):min(b,24)
            Vab[t-max(1,a)+1]=max(0.0,muup[unit.name,t]-mudown[unit.name,t])
        end
    end
end

function compute_heuristic_dual_variable_ramping_down(Wab, price, unit, a,b, phi_a_b, gamma, muup, mudown, current_intervals, status)
    """
    Compute the heuristic dual variable of the ramping up constraint in the matrix Vab
    """
    if b<=24
        for t in max(1,a):min(24,b)
            Wab[t-max(1,a)+1]=max(0.0,sum(price[k]-status*unit.LinearTerm for k in max(1,b-phi_a_b):t-1; init=0))
        end
    else
        for t in max(1,a):min(24,b)
            Wab[t-max(1,a)+1]=0.0
        end
    end
    if min(b,24)-max(1,a)>= unit.Tup+unit.Tdown
        for (c,d) in current_intervals
            for t in max(1,a,c)+unit.Tup*(1-(c==a)):min(24,b,d)-unit.Tdown*(1-(b==d))
                Wab[t-max(1,a)+1]=mudown[unit.name,t]
            end
        end
    elseif abs(1-gamma[unit.name,a,b])<=0.1
        for t in max(1,a):min(24,b)
            Wab[t-max(1,a)+1]=max(0.0,mudown[unit.name,t]-muup[unit.name,t])
        end
    end
end

function heuristic_cost(unit, a, b, price_unit, gamma_val, muup, mudown, current_intervals, status, Vab, Wab)
    """
    Compute the heuristic lower bound of Qjab
    """
    res=0
    T=24
    phiD_a_b=compute_phi_DOWN(unit,a,b)
    phiU_a_b=compute_phi_UP(unit,a,b)
    compute_heuristic_dual_variable_ramping_down(Wab, price_unit, unit, a,b, phiD_a_b, gamma_val, muup, mudown, current_intervals, status)
    compute_heuristic_dual_variable_ramping_up(Vab, price_unit, unit, a,b, phiU_a_b, gamma_val, muup, mudown, current_intervals, status)
    RD=unit.DeltaRampDown
    RU=unit.DeltaRampUp
    Pmin=unit.MinPower
    Pmax=unit.MaxPower
    unitcost=status*unit.LinearTerm
    Pinit=unit.InitialPower
    if a<=1<=b 
        res+=Pinit*(Wab[1]-Vab[1]) #Cost of the initial condition constraint
    end
    if a>=1 #Cost of the start up constraint
        res+=-Vab[1]*Pmin
        res+=Wab[1]*Pmin
    end
    if 1<=b<=T #Cost of the shut down constraint
        res+=-Wab[b-max(1,a)+1]*Pmin
        res+=Vab[b-max(1,a)+1]*Pmin
    end
    for t in max(1,a):min(T,b)
        @inbounds price_t=price_unit[t]
        index=t-max(1,a)+1
        @inbounds Vt=Vab[index]
        @inbounds Wt=Wab[index]
        if b-1>=t>=a+1
            # Cost of ramping constraints
            res+=-Vt*RU
            res+=-Wt*RD
        end
        #Cost of the maximum and minimum capacity constraint
        if t<=min(T-1,b-1)
            z=price_t-unitcost+Vab[index+1]-Vt-Wab[index+1]+Wt
            res+= -max(0,z)*Pmax
            res+=max(0,-z)*Pmin
        elseif t==T && a<=T<=b-1
            z=price_t-unitcost-Vt+Wt
            res+=-max(0,z)*Pmax
            res+=max(0,-z)*Pmin
        end
    end
    return res    
end