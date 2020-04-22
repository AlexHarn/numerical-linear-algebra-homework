import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from tqdm import tqdm

rcParams['figure.figsize'] = 10, 5
rcParams['font.size'] = 15


def construct_A(N, h, p, q):
    A = -p/h**2*(-2*np.identity(N) + np.diag(np.ones(N-1), 1) +
                 np.diag(np.ones(N-1), -1)) + q*np.identity(N)

    A[-1, -2] = 2*A[-1, -2]
    return np.asarray(A, dtype=np.float64)


def qr_iteration_reursive(A, eps=np.finfo(float).eps):
    """
    Implementation of Hessenberg QR algorithm with deflation. This is an
    implementation of Algorithm 28.2 from the textbook, just without shifts
    because that specific shift does not genreally work for symmetric matrices
    and in our case does not converge for large N.
    """
    n, m = A.shape
    if n != m:
        raise ValueError("A has to be a square matrix!")
    if m == 1:
        return [A[0, 0]]
    print(m)  # for status...
    H = np.copy(A)

    while np.min(np.abs(np.diag(H, -1))) > eps:
        Q, R = np.linalg.qr(H)
        H = R@Q

    # find which one triggered the termination
    k = np.argmin(np.abs(np.diag(H, -1)))

    return qr_iteration_reursive(H[:k+1, :k+1]) + \
        qr_iteration_reursive(H[k+1:, k+1:])


def qr_iteration_iterative(A, eps=np.finfo(float).eps):
    """
    Implementation of Hessenberg QR algorithm with deflation. This is an
    implementation of Algorithm 28.2 from the textbook, just without shifts
    because that specific shift does not genreally work for symmetric matrices
    and in our case does not converge for large N.

    This iterative version abuses the fact that here the smallest subdiagonal
    element always happens to be the last one.
    """
    n, m = A.shape
    H = np.copy(A)
    if n != m:
        raise ValueError("A has to be a square matrix!")

    ews = []
    for m in tqdm(range(n, 1, -1)):
        while np.abs(H[m-1, m-2]) > eps:
            Q, R = np.linalg.qr(H)
            H = R@Q
            H[np.abs(H) < np.finfo(float).eps] = 0

        ews.append(H[m-1, m-1])
        H = H[:m-1, :m-1]
    ews.append(H[0, 0])
    return np.asarray(ews)


p = 1
q = 5

# True smallest eigenvalue:
ew_min = p*0.25 + q
print("True smallest ew: {}".format(ew_min))

ews = []
hs = []
for n in range(4, 10):
    print(n)
    N = 2**n
    h = np.pi/N
    A = construct_A(N, h, p, q)
    hs.append(h)
    ews.append(np.min(qr_iteration_iterative(A, eps=1e-6)))
    # ews.append(np.min(qr_iteration_recursive(A, eps=1e-6)))
    print(ews[-1])

ews = np.asarray(ews)
plt.plot(hs, np.abs(ews - ew_min)/hs)
plt.plot(hs, np.abs(ews - ew_min)/hs, 'x')
plt.xlabel(r'$h$')
plt.ylabel(r'$\frac{\lambda^h_{\mathrm{min}} - \lambda_{\mathrm{min}}}{h}$')
plt.tight_layout()
plt.savefig('5.pdf')
