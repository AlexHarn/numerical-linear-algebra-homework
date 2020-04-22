import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5
rcParams['font.size'] = 15


def construct_AB(N, h, p, q):
    B = 1/6*(4*np.identity(N) + np.diag(np.ones(N-1), 1) +
             np.diag(np.ones(N-1), -1))
    B[-1, -2] = 2*B[-1, -2]

    A = -p/h**2*(-2*np.identity(N) + np.diag(np.ones(N-1), 1) +
                 np.diag(np.ones(N-1), -1))

    A[-1, -2] = 2*A[-1, -2]
    return np.asarray(A + q*B), np.asarray(B)


def modified_inverse_iteration(A, B, eps=np.finfo(float).eps):
    """
    Modified implementation of inverse iteration to find the smallest
    eigenvalue of the general eigenvalue problem.
    """
    n, m = A.shape

    # initialize a normalized vector to start with
    v = np.ones(m)/np.sqrt(m)
    new_lam = 1
    old_lam = 0

    while np.abs(new_lam - old_lam) > eps:
        old_lam = new_lam
        w = np.linalg.solve(A, B@v)
        v = w/np.linalg.norm(w)
        new_lam = v.T@B@A@v
        # new_lam = v.T@B.T@A@v <-- Generally, but here B.T = B

    return new_lam


p = 1
q = 5

# True smallest eigenvalue:
ew_min = p*0.25 + q
print("True smallest ew: {}".format(ew_min))

ews = []
hs = []
for n in range(4, 10):
    N = 2**n
    h = np.pi/N
    A, B = construct_AB(N, h, p, q)
    hs.append(h)
    ews.append(modified_inverse_iteration(A, B))
    print(ews[-1])

ews = np.asarray(ews)
plt.plot(hs, np.abs(ews - ew_min)/hs)
plt.plot(hs, np.abs(ews - ew_min)/hs, 'x')
plt.xlabel(r'$h$')
plt.ylabel(r'$\frac{\lambda^h_{\mathrm{min}} - \lambda_{\mathrm{min}}}{h}$')
plt.tight_layout()
plt.savefig('6.pdf')
