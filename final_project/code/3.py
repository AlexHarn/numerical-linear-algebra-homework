import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5
rcParams['font.size'] = 15


def construct_A_simplified(N, p, q):
    h = np.pi/N
    q = h**2/p*q
    A = (2+q)*np.identity(N) - np.diag(np.ones(N-1), 1) - np.diag(np.ones(N-1),
                                                                  -1)
    A[-1, -2] = -2
    return A, h


def construct_A(N, h, p, q):
    A = -p/h**2*(-2*np.identity(N) + np.diag(np.ones(N-1), 1) +
                 np.diag(np.ones(N-1), -1)) + q*np.identity(N)

    A[-1, -2] = 2*A[-1, -2]
    return np.asarray(A, dtype=np.float64)


def inverse_iteration(A, mu, eps=np.finfo(float).eps):
    """
    Implementation of Algorithm 27.2 to find the eigenvalue of A closest to mu.
    """
    n, m = A.shape
    if n != m:
        raise ValueError("A has to be a square matrix!")

    # initialize the normalized vector to start with
    v = np.ones(m)/np.sqrt(m)
    B = A - mu*np.identity(m)
    new_lam = 1
    old_lam = 0

    while np.abs(new_lam - old_lam) > eps:
        old_lam = new_lam
        w = np.linalg.solve(B, v)
        v = w/np.linalg.norm(w)
        new_lam = v.T@A@v

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
    A = construct_A(N, h, p, q)
    hs.append(h)
    # we know A is symmetric and obviously positive definite, therefore
    # choosing mu=0 should always find the smallest ew of A (because all of A's
    # ews are real and positive). Which is then just the inverse power method
    # without shifts.
    ews.append(inverse_iteration(A, 0))
    print(ews[-1])

ews = np.asarray(ews)
plt.plot(hs, np.abs(ews - ew_min)/hs)
plt.plot(hs, np.abs(ews - ew_min)/hs, 'x')
plt.xlabel(r'$h$')
plt.ylabel(r'$\frac{\lambda^h_{\mathrm{min}} - \lambda_{\mathrm{min}}}{h}$')
plt.tight_layout()
plt.savefig('3.pdf')
