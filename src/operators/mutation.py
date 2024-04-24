import numpy as np
from copy import deepcopy
from config import max_init_depth, trunc_mut_rate, shrink_mut_rate, grow_mut_rate, logical_swap_mut_rate, physical_swap_mut_rate
from split_types import swap_dict
from gp_utils import *

# ------------------------------- AGP -------------------------------

def trunc(gp, t_nodes, f_nodes):
    if np.random.uniform() < trunc_mut_rate:
        return
    
    for f_node in f_nodes:
        if f_node.parent is None:
            continue
        if np.random.uniform() < (f_node.depth / max_init_depth):
            f_node.left = None
            f_node.right = None
            init_tree(gp, f_node, gp.X, gp.y, depth_count=f_node.depth, make_terminal=True)

def grow(gp, t_nodes, f_nodes):
    if np.random.uniform() < grow_mut_rate:
        return
    
    for t_node in t_nodes:
        init_tree(gp, t_node, gp.X, gp.y, depth_count=t_node.depth, grow=True)

def logical_swap(gp, t_nodes, f_nodes):
    if np.random.uniform() < logical_swap_mut_rate:
        return

    split_data = [n.f.__defaults__ for n in f_nodes]
    for i in range(len(f_nodes)):
        if f_nodes[i].r2:
            if np.random.uniform() < 1 - f_nodes[i].r2:
                split_type = split_data[i][2].__str__().split(" ")[1]
                f_nodes[i].f = lambda X, A=split_data[i][0], a=split_data[i][1], op=swap_dict[split_type] : op(X[:, A], a)

def physical_swap(gp, t_nodes, f_nodes):
    if np.random.uniform() < physical_swap_mut_rate:
        return

    for i in range(len(f_nodes)):
        if f_nodes[i].r2:
            if np.random.uniform() < 1 - f_nodes[i].r2:
                tmp = f_nodes[i].left
                f_nodes[i].left = f_nodes[i].right
                f_nodes[i].right = tmp

def shrink(gp, t_nodes, f_nodes):
    if np.random.uniform() < shrink_mut_rate:
        return
    
    for f_node in f_nodes:
        if not f_node.parent:
            continue
        parent = f_node.parent
        parent_is_left_child = parent.left == f_node
        parent_is_right_child = parent.right == f_node

        if np.random.uniform() < f_node.depth / max_init_depth:
            if parent_is_left_child:
                if np.random.uniform() < 0.5:
                    parent.left = f_node.left
                    f_node.left.parent = parent
                else:
                    parent.left = f_node.right
                    f_node.right.parent = parent

            elif parent_is_right_child:
                if np.random.uniform() < 0.5:
                    parent.right = f_node.left
                    f_node.left.parent = parent
                else:
                    parent.right = f_node.right
                    f_node.right.parent = parent