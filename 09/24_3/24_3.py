import numpy as np
from scipy.linalg import expm

import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 15

for i in range(50):
    A = np.random.normal(size=(10, 10)) - 2*np.identity(10)
    eigvals = np.linalg.eigvals(A)
    alpha = np.max(np.real(eigvals))
    # if abs(alpha) > 0.1:
        # continue

    ts = np.linspace(0, 20)
    y = []
    for t in ts:
        y.append(np.linalg.norm(expm(t*A)))
    y = np.asarray(y)
    eta = np.exp(ts*alpha)

    plt.semilogy(ts, y, '.', label=r'$\Vert \exp(tA)\Vert_2$')
    plt.semilogy(ts, eta, '.', label=r'$\exp(t\alpha(A))$')
    plt.legend()
    plt.xlim(0, 20)
    plt.xlabel(r'$t$')
    # print()
    # print(np.sum(np.real(eigvals)))
    # print(np.sum(np.imag(eigvals)))
    # print(alpha)
    # print(np.diagonal(A))
    # print(A)

    # plt.show()
    plt.savefig(str(i)+'.pdf')
    plt.clf()
