import numpy as np
from random import choice
from split_types import *
from config import max_init_depth
from operators.selection import *

elementwise_and = lambda *arrays: np.all(arrays, axis=0)

class Node:
    def __init__(self):
        self.type = None
        self.left = None
        self.right = None
        self.parent = None
        self.fitness = None
        self.depth = None
        self.data_mask = None
        self.f = None
        self.r2 = None

def create_terminal(tree):
    tree.f = lambda y_hat : np.median(y_hat)

def init_tree(gp, tree, X, y, depth_count, grow=False, thresh=0.5, make_terminal=False):
    if grow:
        thresh = 1 - (depth_count / max_init_depth)
    tree.depth = depth_count
    if (((depth_count >= max_init_depth) or (np.random.uniform() <= thresh)) and (depth_count > 1)) or make_terminal or len(X) <= 0:
        tree.type = True
        create_terminal(tree)
        return depth_count
    
    tree.type = False
    A = np.random.randint(0, X.shape[1])
    a = choice(X[:, A])
    op = choice(FUNCTIONS)

    tree.f = lambda X, A=A, a=a, op=op : op(X[:, A], a)
    verdicts = tree.f(X)

    l_X, r_X = X[verdicts], X[~verdicts]
    l_y, r_y = y[verdicts], y[~verdicts]
    
    tree.right = Node()
    tree.right.parent = tree
    tree.left = Node()
    tree.left.parent = tree

    if l_X.shape[0] <= 0:
        init_tree(gp, tree.left, l_X, l_y, depth_count + 1, grow)
        return init_tree(gp, tree.right, r_X, r_y, depth_count + 1, grow)
    elif r_X.shape[0] <= 0:
        init_tree(gp, tree.right, r_X, r_y, depth_count + 1, grow)
        return init_tree(gp, tree.left, l_X, l_y, depth_count + 1, grow)
    return max(init_tree(gp, tree.left, l_X, l_y, depth_count + 1, grow), init_tree(gp, tree.right, r_X, r_y, depth_count + 1, grow))

def init_population(gp, population_sz):
    population = np.empty(shape=population_sz, dtype=Node)
    X, y = gp.X, gp.y
    for i in range(population_sz):
        population[i] = Node()
        depth = init_tree(gp, population[i], X, y, depth_count=0)
        population[i].depth = depth
    return population

def init_gp(gp, X_train, y_train, population_sz):
    gp.X = X_train
    gp.y = y_train
    gp.population_sz = population_sz
    gp.population = init_population(gp, population_sz)

def preorder(root, node_list, depth_count=0, include_root=True):
    if root:
        root.depth = depth_count
        if root.parent != None or include_root:
            node_list.append(root)
        preorder(root.left, node_list, depth_count=depth_count + 1, include_root=True)
        preorder(root.right, node_list, depth_count=depth_count + 1, include_root=True)

def update_nodes_list(tree, include_root=True):
    node_list = []
    preorder(tree, node_list, include_root=include_root)
    tnodes = [n for n in node_list if n.type]
    fnodes = [n for n in node_list if not n.type]
    return tnodes, fnodes

def update_population(gp, current_population, offspring, current_fitnesses):
    offspring_fitnesses = gp.evaluate(offspring)
    population = np.hstack((current_population, offspring))
    fitnesses = np.hstack((current_fitnesses, offspring_fitnesses))
    mask1 = np.array(elitism_selection(fitnesses))
    elitism_population = population[mask1]
    remainder_mask = np.array(list(set(range(0, len(population))) - set(mask1)))
    mask2 = np.array(tournament_selection(fitnesses[remainder_mask], gp.population_sz - len(elitism_population)))
    tournament_population = population[mask2]
    return np.hstack((elitism_population, tournament_population))