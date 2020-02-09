import numpy as np


def back_substitute(r, b):
    """
    Applies back substitution to solve the system rx=b with r upper-triangular.

    This is an implementation of Algorithm 17.1.

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
