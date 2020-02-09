import numpy as np
from qr import classical_gram_schmidt_qr
from qr import modified_gram_schmidt_qr
from qr import householder_qr
from solve import back_substitute, forward_substitute


def hw05_02(a, b):
    """
    Solve ax=b using the three different methods for qr factorization.
    """
    print("Solving ax=b with a =")
    print(a)
    print("and b =")
    print(b)
    # if a is well-determined or over-determined we use Algorithm 11.2
    if a.shape[1] <= a.shape[0]:
        print("\nClassical Gram-Schmidt: x =")
        q, r = classical_gram_schmidt_qr(a)
        print(back_substitute(r, q.T@b))

        print("\nModified Gram-Schmidt: x =")
        q, r = modified_gram_schmidt_qr(a)
        print(back_substitute(r, q.T@b))

        print("\nHouseholder Method: x =")
        q, r = householder_qr(a)
        print(back_substitute(r[:a.shape[1]], q[:, :a.shape[1]].T@b))

    # if a is under-determined we do the same thing for a*
    else:
        print("\nClassical Gram-Schmidt: x =")
        q, r = classical_gram_schmidt_qr(a.T)
        print(q@forward_substitute(r.T, b))

        print("\nModified Gram-Schmidt: x =")
        q, r = modified_gram_schmidt_qr(a.T)
        print(q@forward_substitute(r.T, b))

        print("\nHouseholder Method: x =")
        q, r = householder_qr(a.T)
        print(q[:, :a.shape[0]]@forward_substitute(r[:a.shape[0]].T, b))


if __name__ == "__main__":
    print("Part (a)")
    print("--------")
    a = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 7],
                  [4, 2, 3],
                  [4, 2, 2]])
    b = a@np.ones(a.shape[1])
    hw05_02(a, b)
    print("\nPart (b)")
    print("--------")
    a = np.array([[0.7, 0.70711],
                  [0.70001, 0.70711]])
    b = a@np.ones(a.shape[1])
    hw05_02(a, b)

    print("\nPart (c)")
    print("--------")
    a = np.array([[1, 2, 3],
                  [4, 2, 9]])
    b = np.array([6, 15])
    hw05_02(a, b)
