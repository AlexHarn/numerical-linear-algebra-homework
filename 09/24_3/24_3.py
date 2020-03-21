import numpy as np

import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 15

for i in range(10):
    A = np.random.normal(size=(10, 10)) - 2*np.identity(10)
    alpha = np.max(np.real(np.linalg.eigvals(A)))

    t = np.linspace(0, 20)
    y = np.linalg.norm(np.exp(t[:, None, None]*A[None, :, :]), axis=(1, 2))
    eta = np.exp(t*alpha)

    plt.semilogy(t, y, label=r'$\Vert \exp(tA)\Vert_2$')
    plt.semilogy(t, eta, label=r'$\exp(t\alpha(A))$')
    plt.legend()
    plt.xlim(0, 20)
    plt.xlabel(r'$t$')
    # plt.show()
    plt.savefig(str(i)+'.pdf')
    plt.clf()
