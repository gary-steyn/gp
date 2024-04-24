import numpy as np
from gp import GP

def main():
    X = np.random.randint(1, 10, size=[10000, 3])
    f = lambda X: X[:, 0]**2 + X[:, 1] + X[:, 2]
    y = f(X)
    
    gp = GP(X, y, population_sz=100)
    gp.run()
    
if __name__ == "__main__":
    main()