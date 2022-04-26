import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import GAIAfunctions as gf
from scipy import optimize


psi_kappa = 10
i_o_kappa = 10

science_sample = gf.get_binaries(2000, psi_kappa, i_o_kappa)  # Here we can define the distribution of psi and i_o of the hypothetical science sample
control_sample = gf.make_control_sample(science_sample)  # Can take a long time to run


################################## plot the data #############################################
# Assumed orbital inclination distribution of the science sample
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.hist(science_sample['i_o']*180/np.pi, bins=30, density = True)
ax.set_xlabel('Orbital Inclination (degrees)')
plt.show()

#############################################################################################
# Comparison of the measured vbroad of the science sample and control sample
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(science_sample['teff'], science_sample['vbroad'], s=2)
ax.scatter(control_sample['teff'], control_sample['vbroad'], s=2)
ax.set_xlabel('Effective Temperature (K)')
ax.set_ylabel('Rotational Velocity (km/s)')


def f(x):
    mean, a, b, c = x

    hosts = np.sum((science_sample['vbroad'] - mean * gf.v_rotation(science_sample['teff'], a, b, c)) ** 2)
    controls = np.sum((control_sample['vbroad'] - np.pi / 4 * gf.v_rotation(control_sample['teff'], a, b, c)) ** 2)

    return controls + hosts


result = optimize.minimize(f, (1, 5.5, 3.5, 1.8))  # find best order 2 polynomial fit to ALL stars, and best fitting meansini for science stars

T = np.linspace(np.min(science_sample['teff']), np.max(science_sample['teff']), 100)
ax.plot(T, result.x[0]*gf.v_rotation(T, result.x[1], result.x[2], result.x[3]), c='C0', linewidth=5, label='Best fit science sample, <sini> = %1.4f, real <sini> = %1.4f' % (result.x[0], np.mean(np.sin(science_sample['i_s']))))
ax.plot(T, np.pi/4*gf.v_rotation(T, result.x[1], result.x[2], result.x[3]), c='C1', linewidth=5, label='Best fit control stars')
ax.legend()
plt.show()

#%%
#############################################################################################
# vsini method used on the samples: find best fitting polynomial of the entire data set using both science and control sample. Then find by how much higher or lower the mean vsini the science sample is compared to the control sample
mean_sinis = gf.bootstrap_data(1000, control_sample, science_sample)

fig_bootstrap, ax_bootstrap = plt.subplots(1, 1, figsize=(10, 10))
bins_bootstrap = np.linspace(0, 1, 100)
ax_bootstrap.hist(mean_sinis, bins=bins_bootstrap, label='real distribution: $\kappa = %1.0f$, <sini> real = %1.2f, <sini> found = %1.2f' % (psi_kappa, np.mean(np.sin(science_sample['i_s'])), np.mean(mean_sinis)), zorder=10, density = True)
ax_bootstrap.set_xlim(0, 1)


def sini_from_assumed_psi(psi_val=None, psi_kappa=None, psi_m=0, runs=50): # find the sini distribution, given the actual i_o distribution and a few different assumed distributions of psi

    sini_s_means = []
    for ru in range(runs):

        if psi_kappa is not None:
            if psi_kappa == "iso":
                psi = np.arccos(np.random.uniform(0, 1, len(science_sample['i_o'])))
            else:
                psi = gf.vMF_sampler(psi_kappa, psi_m, len(science_sample['i_o']))  # obliquity distribution for which we want to find the corresponding psi

        if psi_val is not None:
            psi = psi_val

        Omega = np.random.uniform(-np.pi * 2, np.pi * 2, len(science_sample['i_o']))  # random placement on the n_s ellipse.
        io = np.random.normal(science_sample['i_o'], science_sample['i_o_err'])  # redraw i_o based on formal uncertainty
        lmbda = np.arctan(np.sin(psi) * np.sin(Omega) / (np.cos(psi) * np.sin(io) + np.sin(psi) * np.cos(Omega) * np.cos(io)))
        i_ss = np.arcsin(np.abs(np.sin(psi) * np.sin(Omega) / np.sin(lmbda)))

        def debug_bootstrap():
            fig, ax = plt.subplots(1, 1, figsize=(
            10, 10))  # checks the i_s and i_o distributions. Only run when "runs" is low
            bins = np.linspace(0, 90, 60)
            ax.hist(science_sample['i_o'] * 180 / np.pi, bins=bins, alpha=0.6, label='i_o science sample')
            ax.hist(science_sample['i_s'] * 180 / np.pi, bins=bins, alpha=0.6, label='i_s science sample, <sini> = %1.2f'%np.mean(np.sin(science_sample['i_s'])), color='red')
            # ax.hist(np.arcsin(np.array(sini_ss)) * 180 / np.pi, bins=bins, alpha=0.6, label='i_s bootstrap sample, <sini> = %1.2f'%np.mean(sini_ss), color='orange')
            ax.hist(science_sample['psi'] * 180 / np.pi, bins=bins, alpha=0.6, label='psi science sample', color='green')
            # ax.hist(psi*180/np.pi, bins=bins, alpha=0.7, label='psi bootstrap', color='blue')
            ax.legend()
            plt.show()

        # debug_bootstrap()

        sini_s_means.append(np.mean(np.sin(i_ss)))
    if psi_kappa is not None and psi_kappa != "iso":
        print("kappa = %1.0f and <sini> = %1.2f" % (psi_kappa, np.mean(sini_s_means)))

    return sini_s_means


ax_bootstrap.hist(sini_from_assumed_psi(psi_kappa="iso"), bins=bins_bootstrap, label='$\psi$ isotropically distributed', alpha=0.7, density=True)
kappa = [3, 6, 10, 20, 60, 200]

for k in kappa:

    sinis = sini_from_assumed_psi(psi_kappa=k)
    ax_bootstrap.hist(sinis, bins=bins_bootstrap, label='$\kappa = %1.0f$' % k, alpha=0.7, density=True)


sinis = sini_from_assumed_psi(psi_kappa=10, psi_m=45*np.pi/180)
ax_bootstrap.hist(sinis, bins=bins_bootstrap, label='$<psi> = 45 deg$', alpha=0.7, density=True)

sinis = sini_from_assumed_psi(psi_kappa=kappa, psi_m=90*np.pi/180)
ax_bootstrap.hist(sinis, bins=bins_bootstrap, label='$<psi> = 90 deg$', alpha=0.7, density=True)

ax_bootstrap.legend()
ax_bootstrap.set_xlabel('<sini>')
plt.show()

#%%
#############################################################################################
# use a kind of numerical "integration" to find the posterior of kappa given the measured mean sini.

a, b = np.histogram(mean_sinis, bins=bins_bootstrap)
kappas = np.linspace(psi_kappa/4, psi_kappa*1.5, 35)

ints = []
for k in kappas:
    sinis = sini_from_assumed_psi(psi_kappa=k, runs=20)
    aa, bb = np.histogram(sinis, bins=bins_bootstrap)

    sums = []
    for count, x in enumerate(a):
        sums.append(a[count] * aa[count])

    ints.append(np.sum(sums))

    def debug_numerical_int():
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.hist(mean_sinis, bins=bins_bootstrap, alpha=0.7, label='real data', density=True)
        ax.hist(sinis, bins=bins_bootstrap, alpha=0.7, label='kappa = 20', density=True)

        x = b[:-1] + (b[1] - b[0]) / 2
        ax.scatter(x, sums)
        ax.set_xlabel('<sini>')
        plt.show()

    # debug_numerical_int()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

prior_kappa = (1+(kappas**2))**(-3/4)
ints = ints*prior_kappa
ints = ints/np.trapz(ints, kappas)
ax.plot(kappas, ints)
ax.set_xlabel('$\kappa$')
ax.set_ylabel('$P(\kappa)$')

plt.show()