import numpy as np
from numpy.linalg import norm


def classical_gram_schmidt_qr(a):
    """
    Compute the qr factorization of a matrix.

    Factor the matrix a as qr, where q is orthonormal and r is
    upper-triangular using the classical Gram-Schmidt method.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to be factored.

    Returns
    -------
    q : ndarray of float or complex, optional
        A matrix with orthonormal columns.

    r : ndarray of float or complex, optional
        The upper-triangular matrix.
    """
    m, n = a.shape
    r = np.zeros((n, n))
    q = np.zeros((m, n))

    for j in range(n):
        v = a[:, j]
        for i in range(j):
            r[i, j] = q[:, i].T@a[:, j]
            v = v - r[i, j]*q[:, i]
        r[j, j] = norm(v)
        q[:, j] = v/r[j, j]
    return q, r


def modified_gram_schmidt_qr(a):
    """
    Compute the qr factorization of a matrix.

    Factor the matrix a as qr, where q is orthonormal and r is
    upper-triangular using the modified Gram-Schmidt method.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to be factored.

    Returns
    -------
    q : ndarray of float or complex, optional
        A matrix with orthonormal columns.

    r : ndarray of float or complex, optional
        The upper-triangular matrix.
    """
    m, n = a.shape
    q = np.zeros((m, n))
    r = np.zeros((n, n))
    v = a.astype(float)

    for i in range(n):
        r[i, i] = norm(v[:, i])
        q[:, i] = v[:, i]/r[i, i]
        for j in range(i+1, n):
            r[i, j] = q[:, i].T@v[:, j]
            v[:, j] = v[:, j] - r[i, j]*q[:, i]

    return q, r


def householder_qr(a):
    """
    Compute the qr factorization of a matrix.

    Factor the matrix a as qr, where q is orthonormal and r is
    upper-triangular using the Householder transform based method.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to be factored.

    Returns
    -------
    q : ndarray of float or complex, optional
        A matrix with orthonormal columns.

    r : ndarray of float or complex, optional
        The upper-triangular matrix.
    """
    m, n = a.shape
    q_t = np.identity(m)
    r = a.astype(float)

    for k in range(n):
        v = np.sign(r[k, k])*norm(r[k:, k])*np.identity(m - k)[0] + r[k:, k]
        v = v/norm(v)
        for i in range(k, n):
            r[k:, i] = r[k:, i] - 2*v*(v.T@r[k:, i])

        # construct Q* using Algorithm 10.3
        for i in range(m):
            q_t[k:, i] = q_t[k:, i] - 2*v*(v.T@q_t[k:, i])

    return q_t.T, r


def eval_qr(a):
    """
    Calculates the QR decomposition A=QR using the 3 different implementations
    as well as the Numpy method and evaluates the results in the way the
    assignment asks for.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to be factored.
    """
    cs_q, cs_r = classical_gram_schmidt_qr(a)
    ms_q, ms_r = modified_gram_schmidt_qr(a)
    h_q, h_r = householder_qr(a)
    np_q, np_r = np.linalg.qr(a)

    # check correctness:
    print("Check for correctness:")
    print("Classical Gram-Schmidt: ||QR - A|| =",
          norm(cs_q@cs_r - a))
    print("Modified Gram-Scmidt: ||QR - A|| =",
          norm(ms_q@ms_r - a))
    print("Householder Transform based: ||QR - A|| =",
          norm(h_q@h_r - a))
    print("Numpy: ||QR - A|| =",
          norm(np_q@np_r - a))

    # check orthogonality
    print('\nCheck orthogonality')
    print("Classical Gram-Schmidt: ||Q*Q - I|| =",
          norm(cs_q.T@cs_q - np.identity(cs_q.shape[1])))
    print("Modified Gram-Scmidt: ||Q*Q - I|| =",
          norm(ms_q.T@ms_q - np.identity(ms_q.shape[1])))
    print("Householder Transform based: ||Q*Q - I|| =",
          norm(h_q.T@h_q - np.identity(h_q.shape[1])))
    print("Numpy: ||Q*Q - I|| =",
          norm(np_q.T@np_q - np.identity(np_q.shape[1])))


if __name__ == "__main__":
    # create the example matrices
    Z = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 7],
                  [4, 2, 3],
                  [4, 2, 2]])
    A = np.array([[0.7, 0.70711],
                  [0.70001, 0.70711]])

    # run the evaluation for both cases
    print('Case 1:\n-------')
    eval_qr(Z)
    print('\nCase 2:\n-------')
    eval_qr(A)
