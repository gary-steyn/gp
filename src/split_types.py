import numpy as np

def lt(x, y):
    return x < y

def gt(x, y):
    return x > y

def eq(x, y):
    return x == y

def ne(x, y):
    return x != y

FUNCTIONS = [lt, gt, eq, ne]

swap_dict = {"eq": ne, "ne": eq, "lt": gt, "gt": lt}