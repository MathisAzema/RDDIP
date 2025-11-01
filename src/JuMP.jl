#  Copyright (c) 2017-25, Oscar Dowson and RDDIP.jl contributors, Lea Kapelevich.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# ============================================================================ #
#
#   Code to implement a JuMP variable extension.
#
#   Usage:
#   julia> @variable(subproblem, 0 <= x[i=1:2] <= i,
#              RDDIP.State, initial_value = i)
#
#   julia> x
#   2-element Array{State{VariableRef},1}:
#     State(x[1]_in,x[1]_out)
#     State(x[2]_in,x[2]_out)
#
#   julia> x[1].in
#   x[1]_in
#
#   julia> typeof(x[1].in)
#   VariableRef
#
#   julia> x[2].out
#   x[2]_out
#
#   Assuming subproblem has been solved, and there exists a primal solution
#   julia> x_values = JuMP.value.(x)
#   2-element Array{State{Float64},1}:
#     State(0.0,1.0)
#     State(1.2,3.0)
#
#   julia> x_values[1].out
#   1.0
# ============================================================================ #

struct StateInfo
    in::JuMP.VariableInfo
    out::JuMP.VariableInfo
    initial_value::Float64
    kwargs::Any
end

struct UncertainInfo
    info::JuMP.VariableInfo
    kwargs::Any
end

function JuMP.build_variable(
    _error::Function,
    info::JuMP.VariableInfo,
    ::Type{State};
    initial_value = NaN,
    kwargs...,
)
    if isnan(initial_value)
        _error(
            "When creating a state variable, you must set the " *
            "`initial_value` keyword to the value of the state variable at" *
            " the root node.",
        )
    end
    return StateInfo(
        JuMP.VariableInfo(
            false,
            NaN,  # lower bound
            false,
            NaN,  # upper bound
            false,
            NaN,  # fixed value
            false,
            NaN,  # start value
            false,
            false, # binary and integer
        ),
        info,
        initial_value,
        kwargs,
    )
end

function JuMP.add_variable(
    subproblem::JuMP.Model,
    state_info::StateInfo,
    name::String,
)
    state = State(
        JuMP.add_variable(
            subproblem,
            JuMP.ScalarVariable(state_info.in),
            name * "_in",
        ),
        JuMP.add_variable(
            subproblem,
            JuMP.ScalarVariable(state_info.out),
            name * "_out",
        ),
    )
    node = get_node(subproblem)
    sym_name = Symbol(name)
    @assert !haskey(node.states, sym_name)  # JuMP prevents duplicate names.
    node.states[sym_name] = state
    graph = get_policy_graph(subproblem)
    graph.initial_root_state[sym_name] = state_info.initial_value
    return state
end

function JuMP.add_variable(
    subproblem::JuMP.Model,
    uncertain_info::UncertainInfo,
    name::String,
)
    v = JuMP.add_variable(
        subproblem,
        JuMP.ScalarVariable(uncertain_info.info),
        name,
    )
    node = get_node(subproblem)
    sym_name = Symbol(name)
    @assert !haskey(node.uncertainties, sym_name)  # prevent duplicate names
    @assert !haskey(node.states, sym_name)  # do not clash with state names
    u = Uncertain(v)
    node.uncertainties[sym_name] = u
    return u
end

function JuMP.build_variable(
    _error::Function,
    info::JuMP.VariableInfo,
    ::Type{Uncertain};
    kwargs...,
)
    # No special initial_value required for uncertain variables.
    return UncertainInfo(info, kwargs)
end

function JuMP.value(state::State{JuMP.VariableRef})
    return State(JuMP.value(state.in), JuMP.value(state.out))
end

function JuMP.value(r::Uncertain{JuMP.VariableRef})
    return Uncertain(JuMP.value(r.var))
end

# Overload for broadcast syntax such as `JuMP.value.([state_1, state_2])`.
Broadcast.broadcastable(state::State{JuMP.VariableRef}) = Ref(state)

# Overload broadcast for Random as well so `JuMP.value.` style works.
Broadcast.broadcastable(r::Uncertain{JuMP.VariableRef}) = Ref(r)

# Allow fixing an Uncertain wrapper by delegating to JuMP.fix on the inner VariableRef
function JuMP.fix(r::Uncertain{JuMP.VariableRef}, value)
    return JuMP.fix(r.var, value)
end

# ==============================================================================

function JuMP.set_optimizer(model::RDDIP.PolicyGraph, optimizer)
    for node in values(model.nodes)
        set_optimizer(node.subproblem, optimizer)
        node.optimizer = optimizer
        set_silent(node.subproblem)
    end
    return
end
