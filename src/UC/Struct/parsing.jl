function parse_nc4(name_instance, optimizer, T=24)
    """
    Parse the SMS++ instances
    """
    data=Dataset(name_instance)
    Blocks=keys(data.group)
    data_block=data.group[Blocks[1]]
    # TimeHorizon= data_block.dim["TimeHorizon"]
    TimeHorizon= T
    demand=data_block["ActivePowerDemand"].var[:]
    N=size(keys(data_block.group))[1]-1
    Thermal_units=Vector{ThermalUnit}(undef, N)
    Hydro_units=Dict()
    for unit in data_block.group
        if occursin("Block", first(unit))
            unit_name=parse(Int64, split(first(unit), "_")[end])+1
            type=last(unit).attrib["type"]
            if type == "ThermalUnitBlock" 
                Bus = 1
                MinPower=last(unit)["MinPower"].var[1]
                MaxPower =last(unit)["MaxPower"].var[1] 
                DeltaRampUp  =last(unit)["DeltaRampUp"].var[1]
                DeltaRampDown  =last(unit)["DeltaRampDown"].var[1]
                QuadTerm =last(unit)["QuadTerm"].var[1] 
                StartUpCost=last(unit)["StartUpCost"].var[1]
                StartDownCost=0.0 
                LinearTerm=last(unit)["LinearTerm"].var[1]
                ConstTerm=last(unit)["ConstTerm"].var[1]
                InitialPower=last(unit)["InitialPower"].var[1]  
                InitUpDownTime =last(unit)["InitUpDownTime"].var[1] 
                MinUpTime=last(unit)["MinUpTime"].var[1]  
                MinDownTime=last(unit)["MinDownTime"].var[1]
                intervals=set_intervals(Int64(MinUpTime), InitUpDownTime, InitialPower, MaxPower, MinPower, DeltaRampDown)
                Tup=Int64(1+floor((MaxPower-MinPower)/DeltaRampUp))
                Tdown=Int64(1+floor((MaxPower-MinPower)/DeltaRampDown))

                unit=ThermalUnit(unit_name, Bus, MinPower, MaxPower, DeltaRampUp, DeltaRampDown, QuadTerm, StartUpCost, StartDownCost, LinearTerm, ConstTerm, InitialPower, InitUpDownTime, Int64(MinUpTime), Int64(MinDownTime), intervals, Tup, Tdown)
                Thermal_units[unit_name]=unit
            end
            if type == "HydroUnitBlock"
                StartArc=last(unit)["StartArc"].var[:]
                EndArc =last(unit)["EndArc"].var[:]
                Inflows  =last(unit)["Inflows"].var[:, :]
                InitialVolumetric  =last(unit)["InitialVolumetric"].var[:]
                MinVolumetric =last(unit)["MinVolumetric"].var[:,:] 
                MaxVolumetric=last(unit)["MaxVolumetric"].var[:,:]  
                UphillFlow=last(unit)["UphillFlow"].var[:]  
                DownhillFlow=last(unit)["DownhillFlow"].var[:]  
                InitialFlowRate=last(unit)["InitialFlowRate"].var[:]  
                DeltaRampUp =last(unit)["DeltaRampUp"].var[:,:] 
                DeltaRampDown=last(unit)["DeltaRampDown"].var[:,:]  
                MinFlow=last(unit)["MinFlow"].var[:,:]
                MaxFlow=last(unit)["MaxFlow"].var[:,:]
                MinPower=last(unit)["MinPower"].var[:,:]
                MaxPower=last(unit)["MaxPower"].var[:,:]
                NumberPieces=last(unit)["NumberPieces"].var[:] 
                LinearTerm=last(unit)["LinearTerm"].var[:]  
                ConstTerm=last(unit)["ConstantTerm"].var[:]
                Hydro_units[unit_name]=HydroUnit(unit_name, StartArc, EndArc, Inflows, InitialVolumetric, MinVolumetric, MaxVolumetric, UphillFlow, DownhillFlow, InitialFlowRate, DeltaRampUp, DeltaRampDown, MinFlow, MaxFlow, MinPower, MaxPower, NumberPieces, LinearTerm, ConstTerm)
            end
        end
    end
    Buses=1:1
    Next=[[]]
    Lines=Dict()
    Demandbus=[demand]
    BusWind=[1]

    training_file = "Data/Uncertainty/training_set.csv"
    table = read_data_forecast(training_file)
    Nb_total_scenario_training=table[end,end]
    Smax=100
    Num_batch=Int(floor(Nb_total_scenario_training/Smax))
    WGscenariob=[]
    for batch in 1:Num_batch
        for day in 1:Smax
            push!(WGscenariob,-table[(batch-1)*Smax*24+(day-1)*24+1:(batch-1)*Smax*24+day*24,1].*demand)
        end
    end
    Training_set=[[(batch-1)*Smax+day for day in 1:Smax] for batch in 1:Num_batch]
    Nb_total_scenario_training=Num_batch*Smax
    
    test_file = "Data/Uncertainty/test_set.csv"
    table = read_data_forecast(test_file)
    Nb_total_scenario_test=table[end,end]
    for day in 1:Nb_total_scenario_test
        push!(WGscenariob,-table[(day-1)*24+1:day*24,1].*demand)
    end
    Test_set=Nb_total_scenario_training+1:Nb_total_scenario_training+Nb_total_scenario_test
    push!(Training_set, [day for day in Nb_total_scenario_training+1:Nb_total_scenario_training+Nb_total_scenario_test])
    WGscenario=[hcat(WGscenariob...)]
    model_Q_jab=Dict{Tuple{Int64, Int64, Int64}, JuMP.Model}()
    return Instance(name_instance, TimeHorizon, N, Thermal_units, Lines, Next, Demandbus, BusWind, WGscenario, Training_set, Test_set, optimizer, model_Q_jab)
end

function parse_IEEE(folder, optimizer)
    """
    Parse the IEEE 118-bus instnace
    """
    TimeHorizon= 24
    syst = "Data/IEEE/"*folder
    generators = CSV.read(joinpath(pwd(), syst, "generators.csv"), DataFrame; header=false)
    NumberUnits= Int(generators[end,1]) 
    N=NumberUnits
    name_instance="IEEE"*string(NumberUnits)
    Thermal_units=Vector{ThermalUnit}(undef, N)
    for i in 1:NumberUnits
        unit_name=i
        Bus = generators[i,2]
        ConstTerm = generators[i,3]
        LinearTerm = generators[i,4]
        MaxPower = generators[i,5]
        MinPower = generators[i,6]
        DeltaRampUp = generators[i,7]
        DeltaRampDown = generators[i,7]
        StartUpCost = generators[i,10]
        StartDownCost = generators[i,10]
        MinUpTime=generators[i,15] 
        MinDownTime=generators[i,15]
        QuadTerm =0.0        
        InitialPower=generators[i,12]
        InitUpDownTime =generators[i,15] *(InitialPower>=1e-3)-generators[i,15]*(InitialPower<=1e-3)
        intervals=set_intervals(MinUpTime, InitUpDownTime, InitialPower, MaxPower, MinPower, DeltaRampDown)
        Tup=Int64(1+floor((MaxPower-MinPower)/DeltaRampUp))
        Tdown=Int64(1+floor((MaxPower-MinPower)/DeltaRampDown))
        unit=ThermalUnit(unit_name, Bus, MinPower, MaxPower, DeltaRampUp, DeltaRampDown, QuadTerm, StartUpCost, StartDownCost, LinearTerm, ConstTerm, InitialPower, InitUpDownTime, MinUpTime, MinDownTime, intervals, Tup, Tdown)
        Thermal_units[unit_name]=unit
    end

    maximum_load = CSV.read(joinpath(pwd(), syst, "maximum_load.csv"), DataFrame; header=false)
    Numbus=maximum_load[end,1]
    Buses=1:Numbus
    load_distribution_profile = CSV.read(joinpath(pwd(), syst, "load_distribution_profile.csv"), DataFrame; header=false)[:,2]/100
    Demandbus=[maximum_load[b,2]*load_distribution_profile for b in Buses]
    df_lines = CSV.read(joinpath(pwd(), syst, "lines.csv"), DataFrame; header=false)
    Numlines=Int(df_lines[end,1])
    Lines=Dict()
    for i in 1:Numlines
        Lines[(Int(df_lines[i,2]), Int(df_lines[i,3]))]=Line(df_lines[i,2],df_lines[i,3], df_lines[i,5], 1/df_lines[i,4])
        Lines[(Int(df_lines[i,3]), Int(df_lines[i,2]))]=Line(df_lines[i,3],df_lines[i,2], df_lines[i,5], 1/df_lines[i,4])
    end
    Next=[[] for b in Buses]
    for i in 1:Numlines
        push!(Next[df_lines[i,2]], df_lines[i,3])
        push!(Next[df_lines[i,3]], df_lines[i,2])
    end
    renewable_generation_profile = Matrix(CSV.read(joinpath(pwd(), syst, "renewable_generation_profile.csv"), DataFrame;header=true))[:,2:end]
    maximum_renewable_generation = CSV.read(joinpath(pwd(), syst, "maximum_renewable_generation.csv"), DataFrame;header=false)
    NumWind,_=size(maximum_renewable_generation)
    Windfarms=1:NumWind
    BusWind=[maximum_renewable_generation[i,1] for i in Windfarms]
    Nb_total_scenario=Int(size(renewable_generation_profile)[1]/24)
    S,  = size(renewable_generation_profile)
    WGscenario=[reshape(renewable_generation_profile[:,b_]*maximum_renewable_generation[b_,2],24,Int(S/24)) for b_ in Windfarms]
    Num_batch=3
    Smax=100
    Training_set=[[(batch-1)*Smax+day for day in 1:Smax] for batch in 1:Num_batch]
    Test_set=Num_batch*Smax+1:Num_batch*Smax+Nb_total_scenario

    model_Q_jab=Dict{Tuple{Int64, Int64, Int64}, JuMP.Model}()
    return Instance(name_instance, TimeHorizon, N, Thermal_units, Lines, Next, Demandbus, BusWind, WGscenario, Training_set, Test_set, optimizer, model_Q_jab)
end