from copy import deepcopy
from random import sample
import numpy as np

k = 10
N = 50

# No. to add
def tournament_selection(fitnesses, N=N):
    selected = []
    for i in range(N):
        candidate_pool = sample(range(0, len(fitnesses)), k)
        selected.append(np.argmin(fitnesses[candidate_pool]))
    if len(selected) > 1:
        return selected
    return selected[0]

def elitism_selection(fitnesses, N=N):
    return np.argsort(fitnesses)[:N]

selection_for_crossover = {"tournament_selection": lambda fitnesses : tournament_selection(fitnesses, 1)}
# selection_for_generation_update = {"tournament_selection": lambda fitnesses : tournament_selection(fitnesses),
#                                     "elitism_selection": lambda fitnesses : elitism(fitnesses)}