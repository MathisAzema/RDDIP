function parse_nc4(name_instance, optimizer, T=24)
    """
    Parse the SMS++ instances
    """
    data=Dataset(name_instance)
    Blocks=keys(data.group)
    data_block=data.group[Blocks[1]]
    # TimeHorizon= data_block.dim["TimeHorizon"]
    TimeHorizon= T
    demand=round.(data_block["ActivePowerDemand"].var[:])
    N=size(keys(data_block.group))[1]-1
    # N=1
    Thermal_units=Vector{ThermalUnit}(undef, N)
    Hydro_units=Dict()
    k=0
    for unit in data_block.group
        if k < N
            if occursin("Block", first(unit))
                k += 1
                unit_name=parse(Int64, split(first(unit), "_")[end])+1
                type=last(unit).attrib["type"]
                if type == "ThermalUnitBlock" 
                    Bus = 1
                    MinPower=round.(last(unit)["MinPower"].var[1])
                    MaxPower =round.(last(unit)["MaxPower"].var[1]) 
                    DeltaRampUp  =round.(last(unit)["DeltaRampUp"].var[1])
                    DeltaRampDown  =round.(last(unit)["DeltaRampDown"].var[1])
                    QuadTerm =round(last(unit)["QuadTerm"].var[1]) 
                    StartUpCost=round(last(unit)["StartUpCost"].var[1])
                    StartDownCost=0.0 
                    LinearTerm=round(last(unit)["LinearTerm"].var[1])
                    ConstTerm=round(last(unit)["ConstTerm"].var[1])
                    InitialPower=round.(last(unit)["InitialPower"].var[1])  
                    InitUpDownTime =round.(last(unit)["InitUpDownTime"].var[1]) 
                    MinUpTime=round.(last(unit)["MinUpTime"].var[1])  
                    MinDownTime=round.(last(unit)["MinDownTime"].var[1])
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

function parse_IEEE_JEAS(folder, optimizer; NumWind=91)
    """
    Parse the IEEE 118-bus instnace
    """
    TimeHorizon= 24
    syst = "Data/"*folder
    generators = CSV.read(joinpath(pwd(), syst, "generators.csv"), DataFrame; header=false)
    NumberUnits= parse(Int64,generators[end,1]) 
    N=NumberUnits
    name_instance="IEEE"*string(NumberUnits)
    Thermal_units=Vector{ThermalUnit}(undef, N)
    for i in 2:NumberUnits+1
        unit_name=i-1
        Bus = parse(Int64,generators[i,2])
        ConstTerm = parse(Float64,generators[i,3])
        LinearTerm = parse(Float64,generators[i,4])
        MaxPower = parse(Float64,generators[i,6])
        MinPower = parse(Float64,generators[i,7])
        DeltaRampUp = parse(Float64,generators[i,14])
        DeltaRampDown = parse(Float64,generators[i,14])
        StartUpCost = parse(Float64,generators[i,15])
        StartDownCost = 0.0*parse(Float64,generators[i,15])
        MinUpTime=parse(Int64,generators[i,13]) 
        MinDownTime=parse(Int64,generators[i,12])
        QuadTerm =0.0        
        InitialPower=parse(Float64,generators[i,11])
        InitUpDownTime =parse(Int64,generators[i,10])
        intervals=set_intervals(Int64(MinUpTime), InitUpDownTime, InitialPower, MaxPower, MinPower, DeltaRampDown)
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
    Numlines=parse(Int64,df_lines[end,1])
    Lines=Vector{Line}(undef, Numlines)
    Next=[[] for b in Buses]
    for i in 2:Numlines+1
        id = parse(Int64, df_lines[i,1])
        b1 = parse(Int64, df_lines[i,2])
        b2 = parse(Int64, df_lines[i,3])
        fmax = parse(Float64, df_lines[i,6])
        X = 1/parse(Float64, df_lines[i,5])
        Lines[i-1]=Line(id, b1, b2, fmax, X)
        push!(Next[b1], b2)
        push!(Next[b2], b1)
    end

    Windfarms=1:NumWind
    BusWind=[b for b in Buses if sum(Demandbus[b])>=1][Windfarms]
    Nb_total_scenario=2000

    WGscenario=[zeros(24,Nb_total_scenario) for b in BusWind]

    Num_batch=10
    Smax=100
    Training_set=[[(batch-1)*Smax+day for day in 1:Smax] for batch in 1:Num_batch]
    Test_set=Num_batch*Smax+1:Nb_total_scenario
    push!(Training_set, [day for day in Test_set])
    model_Q_jab=Dict{Tuple{Int64, Int64, Int64}, JuMP.Model}()

    return Instance(name_instance, TimeHorizon, N, Thermal_units, Lines, Next, Demandbus, BusWind, WGscenario, Training_set, Test_set, optimizer, model_Q_jab)
end