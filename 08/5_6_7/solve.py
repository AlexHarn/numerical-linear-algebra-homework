import numpy as np


def back_substitute(r, b):
    """
    Applies back substitution to solve the system rx=b with r upper-triangular.

    This is an implementation of Algorithm 17.1.

    This implementation ignores any entries below the diagonal in r.

    Parameters
    ----------
    r : array_like, shape (M, M)
        Upper-triangular matrix.
    b : array_like, shape (M)
        Ordinate values.

    Returns
    -------
    x : array_like, shape (M)
        The solution of rx=b.
    """
    m = len(b)
    x = np.zeros(m)
    x[m-1] = b[m-1]/r[m-1, m-1]
    for j in range(m-2, -1, -1):
        x[j] = (b[j] - np.dot(x[j+1:], r[j, j+1:]))/r[j, j]

    return x


def forward_substitute(l, b):
    """
    Applies forward substitution to solve the system lx=b with l
    lower-triangular.

    Parameters
    ----------
    l : array_like, shape (M, M)
        Lower-triangular matrix.
    b : array_like, shape (M)
        Ordinate values.

    Returns
    -------
    x : array_like, shape (M)
        The solution of lx=b.
    """
    m = len(b)
    x = np.zeros(m)
    x[0] = b[0]/l[0, 0]
    for j in range(1, m):
        x[j] = (b[j] - x@l[j])/l[j, j]

    return x


def lu_factorize_inplace(A):
    """
    Constructs the LU decomposition of A and leaves it in place of A such that
    the upper-triangular part of A is U and the entries below the diagonal are
    the entries below the diagonal of L (the diagonal of L is always the one
    vector).

    This is an implemantation of the modified Algorithm 20.1 (Exercise 20.4).
    """
    for k in range(len(A) - 1):
        A[k+1:, k] /= A[k, k]
        A[k+1:, k+1:] -= np.outer(A[k+1:, k], A[k, k+1:])


def gaussian_elimination_no_pivoting(A, b):
    """
    Implementation of Gaussian elimination without pivoting to solve the linear
    system Ax=b.

    Parameters
    ----------
    A : array_like
        The matrix A.
    b : array_like
        The vector b.

    Returns
    -------
    The solution vector x.
    """
    # we can just append b as an additional column to A
    A = np.append(A, np.expand_dims(b, axis=1), axis=1)
    lu_factorize_inplace(A)
    return back_substitute(A[:, :-1], A[:, -1])
