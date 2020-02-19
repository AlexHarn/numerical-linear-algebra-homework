import numpy as np
import matplotlib.pyplot as plt

b = np.float32(1)
c = np.float32(0.004004)

print(1000*(c/(np.sqrt(b**2 + c) - b) - 2*b))
print(1000*c/(np.sqrt(b**2 + c) + b))
