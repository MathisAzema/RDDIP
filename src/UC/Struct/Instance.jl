struct ThermalUnit
    name::Int64
    Bus::Int64
    MinPower::Float64  
    MaxPower::Float64  
    DeltaRampUp::Float64  
    DeltaRampDown::Float64 
    QuadTerm::Float64  
    StartUpCost::Float64
    StartDownCost::Float64  
    LinearTerm::Float64
    ConstTerm::Float64 
    InitialPower::Float64
    InitUpDownTime::Int64
    MinUpTime::Int64  
    MinDownTime::Int64
    intervals::Vector{Vector{Int64}}
    Tup::Int64
    Tdown::Int64
end

struct Line
    b1::Int64
    b2::Int64 
    Fmax::Float64  
    B12::Float64 
end

struct HydroUnit
    name::String
    StartArc::Vector{Int32}
    EndArc::Vector{Int32}
    Inflows::Matrix{Float64}
    InitialVolumetric::Vector{Float64}
    MinVolumetric::Matrix{Float64}
    MaxVolumetric::Matrix{Float64}
    UphillFlow::Vector{Int32}
    DownhillFlow::Vector{Int32}
    InitialFlowRate::Vector{Int32}
    DeltaRampUp::Matrix{Float64}
    DeltaRampDown::Matrix{Float64}
    MinFlow::Matrix{Float64}
    MaxFlow::Matrix{Float64}
    MinPower::Matrix{Float64}
    MaxPower::Matrix{Float64}
    NumberPieces::Vector{Int32}
    LinearTerm::Vector{Float64}
    ConstantTerm::Vector{Float64}
end

struct Instance
    name::String
    TimeHorizon::Int64 
    N::Int64
    Thermalunits::Vector{ThermalUnit}
    Lines::Dict{Tuple{Int64, Int64}, Line}
    Next::Vector{Vector{Int64}}
    Demandbus::Vector{Vector{Float64}}
    BusWind::Vector{Int64}
    WGscenario::Vector{Matrix{Float64}}
    Training_set::Vector{Vector{Int64}}
    Test_set::UnitRange{Int64}
    optimizer::Any
    model_Q_jab::Dict{Tuple{Int64, Int64, Int64}, JuMP.Model}
end