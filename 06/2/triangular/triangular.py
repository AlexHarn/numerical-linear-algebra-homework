import numpy as np
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
from tqdm import tqdm
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5
rcParams['font.size'] = 15

N_MAT = 1000

ms = 2**np.arange(1, 10, dtype=np.int)

rhos = []
norms = []
conds = []
s_mins = []

for m in tqdm(ms):
    mats = np.triu(np.random.normal(scale=np.sqrt(m), size=(N_MAT, m, m)))
    evs = np.linalg.eigvals(mats).flatten()
    svd = np.sort(np.amin(np.linalg.svd(mats, compute_uv=False), axis=1))
    local_conds = np.sort(np.linalg.cond(mats))

    norms.append(np.max(np.linalg.norm(mats, ord=2, axis=(-1, -2))))
    rhos.append(np.max(np.abs(evs)))
    conds.append(local_conds[0])
    s_mins.append(svd[0])

    cdf = ECDF(svd)
    plt.semilogx(cdf.x, cdf.y,)
    plt.xlabel(r'$\sigma_{\mathrm{min}}$')
    plt.ylabel('ECDF')
    plt.xlim(svd[0], svd[-1])
    plt.ylim(0, 1.1)
    # plt.show()
    plt.savefig('./ecdf/{}.pdf'.format(m))
    plt.clf()

    # print(np.argmax(cdf > 0.95))
    # print(len(local_rs), len(cdf))
    # print(cdf[np.argmax(cdf > .95)])
    # print(local_rs[np.argmax(cdf >= .95)])
    # rs.append(local_rs[np.argmax(cdf >= .95)])
    # print(rs[-1])

    # plot
    plt.axhline(color='k')
    plt.axvline(color='k')
    plt.scatter(np.real(evs), np.imag(evs), marker='.', alpha=.5)
    # plt.gca().add_artist(plt.Circle((0, 0), rs[-1]))
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.axis('equal')
    # plt.show()
    plt.savefig('./evs/{}.png'.format(m))
    plt.clf()

rhos = np.asarray(rhos)
norms = np.asarray(norms)
conds = np.asarray(conds)
s_mins = np.asarray(s_mins)

rhos_lin_reg = stats.linregress(ms, rhos)
print('Linear regression results for rho against m:')
print('Slope: {}'.format(rhos_lin_reg[0]))
print('Intercept: {}'.format(rhos_lin_reg[1]))
print('Standard error: {}\n'.format(rhos_lin_reg[-1]))

norms_lin_reg = stats.linregress(ms, norms)
print('Linear regression results for 2-norm against m:')
print('Slope: {}'.format(norms_lin_reg[0]))
print('Intercept: {}'.format(norms_lin_reg[1]))
print('Standard error: {}'.format(norms_lin_reg[-1]))

xs = np.array([-10, 1.1*ms[-1]])
plt.plot(xs, rhos_lin_reg[0]*xs + rhos_lin_reg[1], 'b-')
plt.plot(ms, rhos, 'bx', label=r'$\rho(A)$')

plt.plot(xs, norms_lin_reg[0]*xs + norms_lin_reg[1], 'r-')
plt.plot(ms, norms, 'rx', label=r'$\Vert A\Vert_2$')

plt.plot(ms, s_mins, 'cx', label=r'$\sigma_{\mathrm{min}}$')

plt.xlim(xs[0], xs[-1])
plt.xlabel(r'$m$')
plt.legend()
plt.savefig('norm_spectral.pdf')
plt.xlim(xs[0], 70)
plt.ylim(-5, 200)
plt.savefig('norm_spectral_zoomed.pdf')
plt.clf()

plt.plot(ms, conds, 'gx')
plt.xlim(xs[0], xs[-1])
plt.xlabel(r'$m$')
plt.ylabel(r'$\kappa(A)$')
plt.savefig('cond.pdf')
plt.clf()

plt.plot(ms, s_mins, '.')
plt.xlim(xs[0], xs[-1])
plt.xlabel(r'$m$')
plt.ylabel(r'$\sigma_{\mathrm{min}}$')
plt.savefig('sigma_min.pdf')
