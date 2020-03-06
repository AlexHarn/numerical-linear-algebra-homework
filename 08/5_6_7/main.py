import numpy as np
from scipy.linalg import hilbert
from solve import gaussian_elimination_no_pivoting


if __name__ == "__main__":
    for n in [2, 4, 6, 8, 10]:
        H = hilbert(n)
        b = H@np.ones(n)
        print(gaussian_elimination_no_pivoting(H, b))

