import numpy as np
from scipy.linalg import hilbert
from solve import gaussian_elimination_no_pivoting
from solve import cholesky_factorize
from solve import forward_substitute, back_substitute

import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5
rcParams['font.size'] = 15


if __name__ == "__main__":
    gauss = []
    chol = []
    qr = []
    ns = np.arange(2, 12, 2)
    for n in ns:
        print("n = {}\n----{}".format(n, '-'*len(str(n))))
        H = hilbert(n)
        R = cholesky_factorize(H)
        x_true = np.ones(n)
        b = H@x_true
        q, r = np.linalg.qr(H)
        print("Gaussian Elimination:")
        x = gaussian_elimination_no_pivoting(H, b)
        err = np.linalg.norm(x_true - x)
        gauss.append(err)
        print("x = {}".format(x))
        print('L2-norm(x_true - x) = {}'.format(err))
        print("Cholesky Decomposition:")
        x = back_substitute(R, forward_substitute(R.T, b))
        err = np.linalg.norm(x_true - x)
        chol.append(err)
        print("x = {}".format(x))
        print('L2-norm(x_true - x) = {}'.format(err))
        print("QR Decomposition:")
        x = back_substitute(r, q.T@b)
        err = np.linalg.norm(x_true - x)
        qr.append(err)
        print("x = {}".format(x))
        print('L2-norm(x_true - x) = {}'.format(err))
        print()

    gauss = np.asarray(gauss)
    chol = np.asarray(chol)
    qr = np.asarray(qr)
    plt.semilogy(ns, gauss, '.', label="Gaussian Elimination")
    plt.semilogy(ns, chol, '.', label="Cholesky Decomposition")
    plt.semilogy(ns, qr, '.', label='QR Decomposition')
    plt.legend()
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\Vert x_{\mathrm{True}} - x\Vert_2$")
    # plt.show()
    plt.savefig('7.pdf')
