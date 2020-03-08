import numpy as np
from numpy.linalg import norm

A = np.array([[1., 1.],
              [1., 1.0001],
              [1., 1.0001]])

b = np.array([2, 1e-4, 4.0001])

Ap = np.array([[10.001, -5, -5],
               [-10,     5,  5]])*1e3
P = A@Ap
x = Ap@b
y = P@b

kappa = np.linalg.cond(A)
costheta = norm(y)/norm(b)
theta = np.arccos(costheta)
eta = norm(A)*norm(x)/norm(y)

by = 1/costheta
bx = kappa/(eta*costheta)
Ay = kappa/costheta
Ax = kappa + kappa**2*np.tan(theta)/eta
