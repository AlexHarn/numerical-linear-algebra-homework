import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5
rcParams['font.size'] = 15


def p_coef(x):
    return x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + \
            5376*x**3 - 4608*x**2 + 2304*x - 512


def p_factorized(x):
    return (x - 2)**9


x = np.linspace(-1920, 2080, num=(2080 + 1920)*100, dtype=np.float32)

plt.plot(x, p_coef(x), '.', ms=.5)
plt.xlabel(r'$x$')
plt.ylabel(r'$p_{\mathrm{coef}}(x)$')
# plt.show()
plt.savefig('a.png')
plt.clf()

plt.plot(x, p_factorized(x), '.', ms=.5)
plt.xlabel(r'$x$')
plt.ylabel(r'$p_{\mathrm{fact}}(x)$')
# plt.show()
plt.savefig('b.png')
plt.clf()

plt.plot(x, p_factorized(x) - p_coef(x), '.', ms=.5)
plt.xlabel(r'$x$')
plt.ylabel(r'$p_{\mathrm{fact}}(x) - p_{\mathrm{coef}}(x)$')
# plt.show()
plt.savefig('a_b_diff.png')
plt.clf()
