import numpy as np

A = np.array([[0, 0],
              [0, 0],
              [0, 2]])

print(np.linalg.svd(A))
