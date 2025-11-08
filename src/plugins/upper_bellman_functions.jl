#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

mutable struct Vertex
    value::Float64
    state::Dict{Symbol,Float64}
    variable_ref::Union{Nothing,JuMP.VariableRef}
    variable_refmd::Union{Nothing,JuMP.VariableRef}
end

mutable struct UpperConvexApproximation
    model::JuMP.Model
    variables_ref::Dict{Symbol,JuMP.VariableRef}
    theta::JuMP.VariableRef
    states::Dict{Symbol,JuMP.VariableRef}
    vertices::Vector{Vertex}

    function UpperConvexApproximation(
        model::JuMP.Model,
        variables_ref::Dict{Symbol,JuMP.VariableRef},
        theta::JuMP.VariableRef,
        states::Dict{Symbol,JuMP.VariableRef},
    )
        return new(
            model,
            variables_ref,
            theta,
            states,
            Vertex[],
        )
    end
end

mutable struct UpperBellmanFunction
    global_theta::UpperConvexApproximation
    Lipschitz_constant::Float64
end

function UpperBellmanFunction(
    Lipschitz_constant;
    lower_bound = -Inf,
    upper_bound = Inf,
)
    return InstanceFactory{UpperBellmanFunction}(;
        Lipschitz_constant = Lipschitz_constant,
        lower_bound = lower_bound,
        upper_bound = upper_bound,
    )
end

function initialize_upper_bellman_function(
    factory::RDDIP.InstanceFactory{UpperBellmanFunction},
    model::RDDIP.PolicyGraph{T},
    node::RDDIP.Node{T},
    optimizer
) where {T}
    lower_bound,
    upper_bound,
    Lipschitz_constant = -Inf, Inf, 0.0
    for (kw, value) in factory.kwargs
        if kw == :lower_bound
            lower_bound = value
        elseif kw == :upper_bound
            upper_bound = value
        elseif kw == :Lipschitz_constant
            Lipschitz_constant = value
        end
    end
    if length(node.children) == 0
        lower_bound = upper_bound = 0.0
    end

    # Add epigraph variable for inner approximation
    sp = node.uppersubproblem
    md = JuMP.Model(optimizer)
    set_silent(md)
    Θᴳ = JuMP.@variable(sp)
    Θᴳmd = JuMP.@variable(md)
    @objective(
        md,
        JuMP.objective_sense(sp),
        Θᴳmd,
    )
    primal_obj = JuMP.objective_function(sp)
    JuMP.set_objective_function(
        sp,
        @expression(sp, primal_obj + Θᴳ),
    )
    # Initialize bounds for the objective states. If objective_state==nothing,
    # this check will be skipped by dispatch.
    x′ = Dict(key => var.out for (key, var) in node.states_upper)
    variables_ref = JuMP.@variable(md, xmd[keys(x′)])

    # model convex combination + Lipschitz penalty
    if length(node.children) != 0
                
        JuMP.@variable(sp, δ[keys(x′)])
        JuMP.@variable(sp, δ_abs[keys(x′)] >= 0)
        JuMP.@constraint(sp, [k in keys(x′)], δ_abs[k] >= δ[k])
        JuMP.@constraint(sp, [k in keys(x′)], δ_abs[k] >= -δ[k])

        JuMP.@variable(md, δmd[keys(x′)])
        JuMP.@variable(md, δmd_abs[keys(x′)] >= 0)
        JuMP.@constraint(md, [k in keys(x′)], δmd_abs[k] >= δmd[k])
        JuMP.@constraint(md, [k in keys(x′)], δmd_abs[k] >= -δmd[k])

        JuMP.@variable(sp, 0 <= σ0 <= 1)
        if JuMP.objective_sense(sp) == MOI.MIN_SENSE
            JuMP.@constraint(
                sp,
                theta_cc,
                Lipschitz_constant * sum(δ_abs) + σ0 * upper_bound <= Θᴳ
            )
        else
            println("Not a minimization problem!")
        end

        JuMP.@constraint(sp, x_cc[k in keys(x′)], δ[k] == x′[k])
        JuMP.@constraint(sp, σ_cc, σ0 == 1)

        JuMP.@variable(md, 0 <= σ0md <= 1)
        if JuMP.objective_sense(md) == MOI.MIN_SENSE
            JuMP.@constraint(
                md,
                theta_ccmd,
                Lipschitz_constant * sum(δmd_abs) + σ0md * upper_bound <= Θᴳmd
            )
        else
            println("Not a minimization problem!")
        end

        JuMP.@constraint(md, x_ccmd[k in keys(x′)], δmd[k] == xmd[k])
        JuMP.@constraint(md, σ_ccmd, σ0md == 1)
        # No children => no cost
    else
        JuMP.fix(Θᴳ, 0.0)
        JuMP.fix(Θᴳmd, 0.0)
    end

    return UpperBellmanFunction(
        UpperConvexApproximation(
            md,
            Dict(key => variables_ref[key] for (key, var) in node.states_upper),
            Θᴳ,
            x′
        ), 
        Lipschitz_constant,
    )
end

function _add_vertex(
    V::UpperConvexApproximation,
    θᵏ::Float64,
    xᵏ::Dict{Symbol,Float64}
)
    vertex = Vertex(θᵏ, xᵏ, nothing, nothing)
    _add_vertex_var_to_model(V, vertex)
    push!(V.vertices, vertex)
    return
end

function _add_vertex_var_to_model(V::UpperConvexApproximation, vertex::Vertex)
    model = JuMP.owner_model(V.theta)

    # Add a new variable to the convex combination constraints
    σk = JuMP.@variable(model, lower_bound = 0.0, upper_bound = 1.0)
    JuMP.set_normalized_coefficient(model[:σ_cc], σk, 1)

    xk = vertex.state
    for key in keys(xk)
        JuMP.set_normalized_coefficient(model[:x_cc][key], σk, xk[key])
    end

    vk = vertex.value
    JuMP.set_normalized_coefficient(model[:theta_cc], σk, vk)

    vertex.variable_ref = σk

    md = V.model
    σkmd = JuMP.@variable(md, lower_bound = 0.0, upper_bound = 1.0)
    JuMP.set_normalized_coefficient(md[:σ_ccmd], σkmd, 1)

    xk = vertex.state
    for key in keys(xk)
        JuMP.set_normalized_coefficient(md[:x_ccmd][key], σkmd, xk[key])
    end

    vk = vertex.value
    JuMP.set_normalized_coefficient(md[:theta_ccmd], σkmd, vk)

    vertex.variable_refmd = σkmd
    return
end

function refine_bellman_function_upper(
    model::PolicyGraph{T},
    node::Node{T},
    upper_bellman_function::UpperBellmanFunction,
    outgoing_state::Dict{Symbol,Float64},
    objective_realization::Float64,
) where {T}
    lock(node.lock)
    try
        return _refine_upper_bellman_function_no_lock(
            model,
            node,
            upper_bellman_function,
            outgoing_state,
            objective_realization,
        )
    finally
        unlock(node.lock)
    end
end

function _refine_upper_bellman_function_no_lock(
    model::PolicyGraph{T},
    node::Node{T},
    upper_bellman_function::UpperBellmanFunction,
    outgoing_state::Dict{Symbol,Float64},
    objective_realization::Float64,
) where {T}
    # The meat of the function.
    return _add_average_vertex(
        node,
        outgoing_state,
        objective_realization,
    )

end

function _add_average_vertex(
    node::Node,
    outgoing_state::Dict{Symbol,Float64},
    objective_realization::Float64,
)
    θᵏ = objective_realization
    _add_vertex(
        node.upper_bellman_function.global_theta,
        θᵏ,
        outgoing_state,
    )
    return (theta = θᵏ, x = outgoing_state)
end

function compute_upper_bellman_value(
    bellman_function::UpperBellmanFunction,
    state::Dict{Symbol,Float64},
)
    V = bellman_function.global_theta
    md = V.model

    # Set the state values in the model
    for (key, value) in state
        JuMP.fix(V.variables_ref[key], value; force = true)
    end

    # Optimize the model
    JuMP.optimize!(md)

    # Retrieve the optimal value
    return JuMP.objective_value(md)
end

function compute_lower_bellman_value(
    bellman_function::BellmanFunction,
    state::Dict{Symbol,Float64},
)
    V = bellman_function.global_theta
    value = maximum([cut.intercept + sum(cut.coefficients[i] * state[i] for (i, x) in V.states) for cut in V.cuts]; init = 0.0)

    # Retrieve the optimal value
    return value
end

function get_worst_case_scenario_by_enumeration(    
    model::PolicyGraph{T},
    node::Node{T},
    state::Dict{Symbol,Float64},
    duality_handler::Union{Nothing,AbstractDualityHandler};
    refine_upper_bound::Bool = true, #true = backward
) where {T}
    objectives = [0.0 for noise in node.noise_terms]
    for (i, noise) in enumerate(node.noise_terms)
        @_timeit_threadsafe model.timer_output "solve_subproblem" begin
            subproblem_results = solve_subproblem_upper(
                model,
                node,
                state,
                noise.term,
                duality_handler = duality_handler
            )
            objectives[i] = subproblem_results.objective
        end
    end
    imax = argmax(objectives)
    if node.index > 1 && refine_upper_bound
        previous_node = model.nodes[node.index-1]
        refine_bellman_function_upper(
            model,
            previous_node,
            previous_node.upper_bellman_function,
            state,
            objectives[imax],
        )
    end

    return (objective = objectives[imax], noise = node.noise_terms[imax].term)
end


function get_worst_case_scenario_by_lagrangian(    
    model::PolicyGraph{T},
    node::Node{T},
    state::Dict{Symbol,Float64},
    outgoing_stage_heur::Dict{Symbol,Float64};
    refine_upper_bound::Bool = true, #true = backward
) where {T}
    md = node.lagrangian_upper.model
    primal_obj = JuMP.objective_function(md)

    JuMP.set_objective_function(
        md,
        @expression(md, primal_obj + sum(
            node.lagrangian_upper.dual_variables_state_in[name] * state[name] for
            (name, _) in node.states)),
    )

    cstr = node.lagrangian_upper.upper_constraint
    for (name, value) in outgoing_stage_heur
        var = node.lagrangian_upper.dual_variables_state_out[name]
        set_normalized_coefficient(cstr, var, -value)
    end

    V_x = compute_upper_bellman_value(
        node.upper_bellman_function,
        outgoing_stage_heur,
    )
    set_normalized_rhs(cstr, V_x)

    JuMP.optimize!(md)

    if JuMP.termination_status(md) != MOI.OPTIMAL
        set_optimizer_attribute(md, "Presolve", 0)
        JuMP.optimize!(md)
    end
    if JuMP.termination_status(md) != MOI.OPTIMAL && JuMP.termination_status(md) != MOI.DUAL_INFEASIBLE
        println(md)
        ps = JuMP.primal_status(md)
        ds = JuMP.dual_status(md)
        println("Primal status: ", ps)
        println("Dual status: ", ds)
        println(objective_value(md))
        println((node.index, length(node.lagrangian_upper.cuts)))
        # end
        return nothing, nothing
    end

    noise = Dict{Symbol,Float64}()
    for (name, var) in node.lagrangian_upper.uncertainty
        noise[name] = JuMP.value(var)
    end
    objective = JuMP.objective_value(md)

    num_states = length(node.states_upper)
    λ = zeros(num_states)
    h_expr = Vector{AffExpr}(undef, num_states)
    for (i, (key, state)) in enumerate(node.states_upper)
        λ[i] = value(node.lagrangian_upper.dual_variables_state_in[key])
        # JuMP.unfix(state.in) #pas besoin apparemment
        h_expr[i] = @expression(node.uppersubproblem, state.in)
    end

    num_uncertainties = length(node.uncertainties_upper)
    μ = zeros(num_uncertainties)
    g_expr = Vector{AffExpr}(undef, num_uncertainties)
    for (i, (key, uncertainty)) in enumerate(node.uncertainties_upper)
        μ[i] = value(node.lagrangian_upper.dual_variables_uncertainty[key])
        # JuMP.unfix(uncertainty.var)
        g_expr[i] = @expression(node.uppersubproblem, uncertainty.var)
    end

    JuMP.set_objective_function(md, primal_obj)

    lagrangian_obj, cost_to_go_value, incoming_state_values, incoming_uncertainty_values, outgoing_state_values = _solve_primal_problem_upper(node, node.uppersubproblem, λ, h_expr, μ, g_expr)

    intercept = lagrangian_obj + sum(λ[i] * incoming_state_values[k] for (i, k) in enumerate(keys(node.states_upper))) +
        sum(μ[i] * incoming_uncertainty_values[k] for (i, k) in enumerate(keys(node.uncertainties_upper)))
    
    _refine_lagrangian_model(
        node,
        intercept,
        cost_to_go_value,
        incoming_state_values,
        incoming_uncertainty_values,
        outgoing_state_values,
        1,
    )

    cost_to_go_value_lower = compute_lower_bellman_value(
        node.bellman_function,
        outgoing_state_values,
    )
    if cost_to_go_value_lower >= cost_to_go_value + 1e-5 
        println("Lower bound value is greater than upper bound value!")
    end
    intercept_lower = intercept - cost_to_go_value + cost_to_go_value_lower
    _refine_lagrangian_model(
        node,
        intercept_lower,
        cost_to_go_value_lower,
        incoming_state_values,
        incoming_uncertainty_values,
        outgoing_state_values,
        0,
    )

    # println(incoming_state_values)


    if node.index > 1 && refine_upper_bound
        previous_node = model.nodes[node.index-1]
        refine_bellman_function_upper(
            model,
            previous_node,
            previous_node.upper_bellman_function,
            state,
            objective,
        )
    end

    if node.index == 1 
        println(objective)
    end

    return (objective = objective, noise = noise)
end