from config import n_selection, k_tourn_sel
from random import sample
import numpy as np
from math import floor

def tournament_selection(fitnesses, N=n_selection):
    selected = []
    for i in range(N):
        candidate_pool = sample(range(0, len(fitnesses)), k_tourn_sel)
        selected.append(np.argmin(fitnesses[candidate_pool]))
    if len(selected) > 1:
        return selected
    return selected[0]

def elitism_selection(fitnesses, N=n_selection):
    return np.argsort(fitnesses)[:N]

selection_for_crossover = {"tournament_selection": lambda fitnesses : tournament_selection(fitnesses, 1)}