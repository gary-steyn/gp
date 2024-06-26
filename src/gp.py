import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import random
from random import shuffle
from split_types import *
from gp_utils import *
from operators.mutation import *
from operators.selection import *
from operators.crossover import *
from config import n_generations, global_mut_rate, global_xo_rate
from sklearn.metrics import r2_score

class GP:
    def __init__(self, X_train, y_train, population_sz, seed=0):
        np.random.seed(seed)
        random.seed(seed)
        init_gp(self, X_train, y_train, population_sz)

    def eval(self, tree, X, y):
        if X.shape[0] <= 0:
            return []
        if tree.type is True:
            y_pred = [tree.f(y)] * y.size
            tree.r2 = max(0, r2_score(y, y_pred))
            return list(zip(y, y_pred))
        verdicts = tree.f(X)
        l_X, r_X = X[verdicts], X[~verdicts]
        l_y, r_y = y[verdicts], y[~verdicts]
        output = self.eval(tree.left, l_X, l_y) + self.eval(tree.right, r_X, r_y)
        tmp = np.array(output)
        tree.r2 = max(0, r2_score(tmp[:, 0], tmp[:, 1]))
        return output
    
    def evaluate(self, offspring=None):
        X, y = self.X, self.y
        if offspring is not None:
            population = offspring
        else:
            population = self.population
        fitnesses = np.zeros(len(population))
        for i, tree in enumerate(population):
            res = np.array(self.eval(tree, X, y))
            fitnesses[i] = np.mean((res[:, 0] - res[:, 1])**2, axis=0) 
        return fitnesses

    def run(self):
        for n in range(n_generations):
            offspring = []
            self.fitnesses = self.evaluate()
            for i, tree in enumerate(self.population):
                if np.random.uniform() < global_mut_rate:
                    self.mutate(tree)
                if np.random.uniform() < global_xo_rate:
                    child = crossover(self, tree)
                    offspring.append(child)
            self.population = update_population(self, self.population, offspring, self.fitnesses)

    def mutate(self, tree):
        tnodes, fnodes = update_nodes_list(tree)
        
        mutations = [trunc, shrink, grow, logical_swap, physical_swap]
        shuffle(mutations)

        for apply_mutation in mutations:
                apply_mutation(self, tnodes, fnodes)
                tnodes, fnodes = update_nodes_list(tree)
        
        print(len(tnodes), len(fnodes))