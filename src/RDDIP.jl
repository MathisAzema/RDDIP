#  Copyright (c) 2017-25, Oscar Dowson and RDDIP.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

module RDDIP

import Reexport
Reexport.@reexport using JuMP

import Distributed
import HTTP
import JSON
import MutableArithmetics
import Printf
import Random
import SHA
import Statistics
import TimerOutputs

using Gurobi
using NCDatasets
using XLSX
using CSV
using DataFrames

# Work-around for https://github.com/JuliaPlots/RecipesBase.jl/pull/55
# Change this back to `import RecipesBase` once the fix is tagged.
using RecipesBase

export @stageobjective

# Modelling interface.
include("user_interface.jl")
include("modeling_aids.jl")

# Default definitions for RDDIP related modular utilities.
include("plugins/headers.jl")

# Tools for overloading JuMP functions
include("binary_expansion.jl")
include("JuMP.jl")

# Printing utilities.
include("cyclic.jl")
include("print.jl")

# The core RDDIP code.
include("algorithm.jl")

# Specific plugins.
include("plugins/risk_measures.jl")
include("plugins/sampling_schemes.jl")
include("plugins/bellman_functions.jl")
include("plugins/stopping_rules.jl")
include("plugins/local_improvement_search.jl")
include("plugins/duality_handlers.jl")
include("plugins/parallel_schemes.jl")
include("plugins/backward_sampling_schemes.jl")
include("plugins/forward_passes.jl")

# Visualization related code.
include("visualization/publication_plot.jl")
include("visualization/spaghetti_plot.jl")
include("visualization/dashboard.jl")
include("visualization/value_functions.jl")

# Other solvers.
include("deterministic_equivalent.jl")
include("biobjective.jl")
include("alternative_forward.jl")

# Inner approximation
include("Inner.jl")

include("Experimental.jl")
include("MSPFormat.jl")

#UC 
const SHEDDING_COST=700.0
const CURTAILEMENT_COST=700.0

include("UC/Struct/Instance.jl")
include("UC/Struct/tools.jl")
include("UC/Unit/Thermal_unit.jl")
include("UC/Unit/Line.jl")
include("UC/Struct/parsing.jl")
include("UC/Optimizer/extensive_formulation.jl")
include("UC/Optimizer/initialisation_Benders.jl")
include("UC/Optimizer/second_stage_SP.jl")
include("UC/Optimizer/add_cut_SP.jl")
include("UC/Optimizer/add_cut_AVAR.jl")
include("UC/Optimizer/benders.jl")
include("UC/Optimizer/benders_AVAR.jl")
include("UC/Optimizer/options.jl")
end
