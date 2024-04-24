import numpy as np

# a = lambda x=2 : x**2
# print(a.__str__())

A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

C = np.hstack((A, B))

print(np.argsort(C))
