import numpy as np

A = np.array([[1, -1],
              [0, 0],
              [2, 1],
              [0, 2]])
q, r = np.linalg.qr(A)
q = -q
r = -r
print("q = ")
print(q)
print("r = ")
print(r)

b = np.array([0, 0, 3, 2])
print("Q* b =")
y = q.T@b
print(y)

print("Solution x = ")
print(np.linalg.solve(r, y))

print("check: A+ =")
Ap = np.linalg.inv(A.T@A)@A.T
print(Ap)
print("x = ")
print(Ap@b)
