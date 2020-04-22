import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5
rcParams['font.size'] = 15

p = 1
q = 5

# True smallest eigenvalue:
ew_min = p*0.25 + q
print("True smallest ew: {}".format(ew_min))


def construct_A_simplified(N, h, p, q):
    A = -p/h**2*(-2*np.identity(N) + np.diag(np.ones(N-1), 1) +
                 np.diag(np.ones(N-1), -1)) + q*np.identity(N)

    A[-1, -2] = 2*A[-1, -2]
    return A, h


def construct_A(N, h, p, q):
    A = -p/h**2*(-2*np.identity(N) + np.diag(np.ones(N-1), 1) +
                 np.diag(np.ones(N-1), -1)) + q*np.identity(N)

    A[-1, -2] = 2*A[-1, -2]
    return np.asarray(A, dtype=np.float64)


def shifted_power(A, sigma, eps=np.finfo(float).eps):
    """
    Implementation of the shifted power method.
    """
    n, m = A.shape
    if n != m:
        raise ValueError("A has to be a square matrix!")

    # initialize the normalized vector to start with
    v = np.ones(m)/np.sqrt(m)
    B = A - sigma*np.identity(m)
    new_lam = 1
    old_lam = 0

    while np.abs(new_lam - old_lam) > eps:
        old_lam = new_lam
        v = B@v
        v = v/np.linalg.norm(v)
        new_lam = v.T@A@v

    return new_lam


ews = []
hs = []
for n in range(4, 10):
    print("n = {}".format(n))
    N = 2**n
    h = np.pi/N
    A = construct_A(N, h, p, q)
    hs.append(h)
    # we want to find the smallest eigenvalue, so the ideal shift should be as
    # close as possible to the largest eigenvalue, so let's find that one first
    lam_max = shifted_power(A, 0)
    print(lam_max)
    # and now use it as a shift to find the smallest
    ews.append(shifted_power(A, lam_max))
    print(ews[-1])
    print("")

ews = np.asarray(ews)
plt.plot(hs, np.abs(ews - ew_min)/hs)
plt.plot(hs, np.abs(ews - ew_min)/hs, 'x')
plt.xlabel(r'$h$')
plt.ylabel(r'$\frac{\lambda^h_{\mathrm{min}} - \lambda_{\mathrm{min}}}{h}$')
plt.tight_layout()
plt.savefig('4.pdf')
