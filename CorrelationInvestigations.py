import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import GAIAfunctions as gf
from scipy import optimize


def moving_average(x, y, size):  # calculate moving average of sin i

    N = len(y)
    y = y[np.argsort(x)]
    avg = np.zeros(N)

    for i in range(N):
        start = max(0, i - size)
        end = min(N, i + size)

        avg[i] = np.mean(y[start:end])

    return np.sort(x), avg


psi_kappa = 10
i_o_kappa = 470

science_sample = gf.get_binaries(500, psi_kappa, i_o_kappa, dependencies=True)  # Here we can define the distribution of psi and i_o of the hypothetical science sample
control_sample = gf.make_control_sample(science_sample)  # Can take a long time to run
#%%
# plot vsini as a function of teff

def f_only_controls(x):
    mean, a, b, c = x

    return np.sum((control_sample['vbroad'] - np.pi / 4 * gf.v_rotation(control_sample['teff'], a, b, c)) ** 2)


result = optimize.minimize(f_only_controls, (1, 5.5, 3.5, 1.8))  # find best order 2 polynomial fit to ALL stars, and best fitting meansini for science stars

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(science_sample['teff'], science_sample['vbroad'], s=4, zorder=10)
ax.scatter(control_sample['teff'], control_sample['vbroad'], s=2)

T = np.linspace(control_sample['teff'].min(), control_sample['teff'].max(), 100)
plt.plot(T, gf.v_rotation(T, result.x[1], result.x[2], result.x[3]), 'k-')
plt.show()


# plot vsini_measured / vsini_calculated (i.e. apparent sini)

sinis = science_sample['vbroad']/gf.v_rotation(science_sample['teff'], result.x[1], result.x[2], result.x[3])


def moving_average_plot(param):
    a, b = moving_average(param, sinis, 40)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(param, sinis)
    ax.plot(a, b, 'k-', linewidth=3)
    ax.set_ylabel('$\sin i$')

    return fig, ax


def threshold_plot(param):
    a_s = np.linspace(np.min(param), np.max(param), 100)
    a_s = a_s[1:-2]

    rels = []
    for a in a_s:
        mask = param > a
        rels.append(np.mean(sinis[~mask]) / np.mean(sinis[mask]))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(a_s, rels)
    ax.set_ylabel('above x / below x')

    return fig, ax


fig, ax = moving_average_plot(science_sample['a_over_R'])
ax.set_xlabel('$a/R$')
plt.show()

fig, ax = threshold_plot(science_sample['a_over_R'])
ax.set_xlabel('$a/R$')
plt.show()


mask = science_sample['a_over_R'] > 200
mean_sinis1 = gf.bootstrap_data(100, control_sample, science_sample[mask])
mean_sinis2 = gf.bootstrap_data(100, control_sample, science_sample[~mask])

fig_bootstrap, ax_bootstrap = plt.subplots(1, 1, figsize=(10, 10))
bins_bootstrap = np.linspace(0, 1, 100)

ax_bootstrap.hist(mean_sinis1, bins=bins_bootstrap, density=True)
ax_bootstrap.hist(mean_sinis2, bins=bins_bootstrap, density=True)

ax_bootstrap.set_xlim(0, 1)
plt.show()
