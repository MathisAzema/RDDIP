"""
H: Heuristic 
E: Exact Qjab
O: Optimal cut
F: feasibility cut
"""

extended_BD_FH_OH = UCOptions(
    master_SP_problem,
    oracle_SP_problem,
    second_stage_SP_extended,
    add_cut_SP,
    _add_feasibility_cuts_extended_H,
    _add_optimality_cuts_extended_H
)

binBD = UCOptions(
    master_3BD_problem,
    oracle_SP_problem,
    second_stage_SP_3bin,
    add_cut_3bin,
    _add_feasibility_cuts_3bin,
    _add_optimality_cuts_3bin
)

extended_BD_FH_OE = UCOptions(
    master_SP_problem,
    oracle_SP_problem,
    second_stage_SP_extended,
    add_cut_SP,
    _add_feasibility_cuts_extended_H,
    _add_optimality_cuts_extended_exact
)

extended_BD_H_AVAR = UCOptions(
    master_AVAR_problem,
    oracle_SP_problem,
    second_stage_SP_extended,
    add_cut_AVAR,
    _add_feasibility_cuts_extended_H,
    _add_optimality_cuts_extended_AVAR
)

extended_BD_FH_OH_multi = UCOptions(
    master_SP_problem,
    oracle_SP_problem,
    second_stage_SP_extended,
    add_cut_SP,
    _add_feasibility_cuts_extended_H,
    _add_multi_optimality_cuts_extended_H
)