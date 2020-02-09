import numpy as np
from scipy.linalg import hilbert
from solve import back_substitute
from qr import householder_qr

import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 10
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 15


def solve_hilbert(n, alpha):
    """
    Solves Ax=b with L2 regularization where A is the n-th order Hilbert
    matrix and b is taken such that the true solution is the one vector.

    Parameters
    ----------
    n : integer
        Order of the Hilbert matrix.
    alpha : float
        The Regularization strength factor.

    Returns
    -------
    The the 2-norm of the difference between the true and reconstructed
    solution.
    """
    h = hilbert(n)
    a = np.vstack((h, alpha*np.identity(n)))
    b = np.append(h@np.ones(n), np.zeros(n))
    q, r = householder_qr(a)
    return np.linalg.norm(np.ones(n) -
                          back_substitute(r[:a.shape[1]],
                                          q[:, :a.shape[1]].T@b))


if __name__ == "__main__":
    v_solve_hilbert = np.vectorize(solve_hilbert)
    alphas = np.logspace(-15, 1, num=150)
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    for i, n in enumerate([10, 20, 30]):
        print("n = {}: error = {} for alpha = 0.".format(n, solve_hilbert(n,
                                                                          0)))
        axs[i].loglog(alphas, v_solve_hilbert(n, alphas))
        axs[i].set_title(r'$n={}$'.format(n))
        axs[i].set_ylabel(r'$\left\Vert x_{\mathrm{True}} - '
                          r'x_{\mathrm{Solved}}\right\Vert_2$')
    plt.xlabel(r'$\alpha$')
    plt.savefig('hw05_3.pdf')
