from random import choice
from copy import deepcopy
from gp_utils import update_nodes_list
from operators.selection import selection_for_crossover

selection_method = "tournament_selection"

def crossover(gp, tree):
    offspring = deepcopy(tree)
    # selected = choice(gp.population)
    selected = gp.population[selection_for_crossover[selection_method](gp.fitnesses)]
    selected_t_nodes, selected_f_nodes = update_nodes_list(selected, include_root=False)

    tree_t_nodes, tree_f_nodes = update_nodes_list(offspring, include_root=False)

    exchanged_tree_node = choice(tree_t_nodes + tree_f_nodes)
    exchanged_selected_node = deepcopy(choice(selected_t_nodes + selected_f_nodes))

    parent = exchanged_tree_node.parent
    is_left_child = parent.left == exchanged_selected_node
    if not is_left_child:
        parent.right = exchanged_selected_node
    else:
        parent.left = exchanged_selected_node
    exchanged_selected_node.parent = parent

    # NOTE: Following function does redundant computation
    update_nodes_list(offspring)
    return offspring