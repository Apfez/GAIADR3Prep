import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import GAIAfunctions as gf
from scipy import optimize

# Get all stars from csv for use as control sample
control_stars_unfiltered = pd.read_csv("all_gaia_stars-result.csv")

# Add some noise to the data. This is only necessary when using simulated gaia results
control_stars_unfiltered['teff'] = control_stars_unfiltered['teff'] + np.random.uniform(-100, 100,
                                                                                        len(control_stars_unfiltered))
control_stars_unfiltered['logg'] = control_stars_unfiltered['logg'] + np.random.uniform(-0.1, 0.1,
                                                                                        len(control_stars_unfiltered))
control_stars_unfiltered['feh'] = control_stars_unfiltered['feh'] + np.random.uniform(-0.1, 0.1,
                                                                                      len(control_stars_unfiltered))
psi_err_real = 20*np.pi/180
i_o_err_real = 15*np.pi/180

science_sample = gf.get_binaries(500, psi_err=psi_err_real, i_o_err=i_o_err_real)  # Here we can define the distribution of psi and i_o of the hypothetical science sample
control_sample = gf.make_control_sample(control_stars_unfiltered, science_sample)  # Can take a long time to run

################################## plot the data #############################################

# Assumed orbital inclination distribution of the science sample
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.hist(science_sample['i_o']*180/np.pi, bins=30)
plt.show()


# Comparison of the measured vbroad of the science sample and control sample
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(science_sample['teff'], science_sample['vbroad'], s=2)
ax.scatter(control_sample['teff'], control_sample['vbroad'], s=2)


def f(x):
    mean, a, b, c = x

    hosts = np.sum((science_sample['vbroad'] - mean * gf.v_rotation(science_sample['teff'], a, b, c)) ** 2)
    controls = np.sum((control_sample['vbroad'] - np.pi / 4 * gf.v_rotation(control_sample['teff'], a, b, c)) ** 2)

    return controls + hosts


result = optimize.minimize(f, (1, 5.5, 3.5, 1.8))  # find best order 2 polynomial fit to ALL stars, and best fitting meansini for science stars

T = np.linspace(np.min(science_sample['teff']), np.max(science_sample['teff']), 100)
ax.plot(T, result.x[0]*gf.v_rotation(T, result.x[1], result.x[2], result.x[3]), c='C0', linewidth=5)
ax.plot(T, np.pi/4*gf.v_rotation(T, result.x[1], result.x[2], result.x[3]), c='C1', linewidth=5)
plt.show()


# vsini method used on the samples: find best fitting polynomial of the entire data set using both science and control sample. Then find by how much higher or lower the mean vsini the science sample is compared to the control sample

runs = 100

mean_vsinis = []
for ru in range(runs):
    # randomly redrawing from the same distribution ala louden

    vsinis_control_new = np.array(control_sample['vbroad'])[np.random.randint(0, len(control_sample['vbroad']), len(control_sample['vbroad']))]  # redrawing all control stars with repetitions allowed
    Ts_control_new = np.array(control_sample['teff'])[np.random.randint(0, len(control_sample['vbroad']), len(control_sample['vbroad']))]

    def bootstrap_mean_is(Ts, vsinis):

        vsinis_science_new = np.array(vsinis)[np.random.randint(0, len(vsinis), len(vsinis))]  # redrawing all science stars with repetitions allowed
        Ts_science_new = np.array(Ts)[np.random.randint(0, len(vsinis), len(vsinis))]

        def f(x):  # eq 7 from louden
            mean, a, b, c = x

            hosts = np.sum((vsinis_science_new - mean * gf.v_rotation(Ts_science_new, a, b, c)) ** 2)
            controls = np.sum((vsinis_control_new - np.pi / 4 * gf.v_rotation(Ts_control_new, a, b, c)) ** 2)

            return controls + hosts

        result = optimize.minimize(f, (1, 5.5, 3.5, 1.8))  # find best order 2 polynomial fit to ALL stars, and best fitting meansini for science stars

        return result.x[0]

    mean_vsinis.append(bootstrap_mean_is(science_sample['teff'], science_sample['vbroad']))

fig_bootstrap, ax_bootstrap = plt.subplots(1, 1, figsize=(10, 10))

bins_bootstrap = np.linspace(0, 1, 100)
ax_bootstrap.hist(mean_vsinis, bins=bins_bootstrap, label='mean sini from measurements ($\sigma\psi = %1.1f^\circ$)'%(psi_err_real*180/np.pi), zorder=10)
ax_bootstrap.set_xlim(0, 1)

# find the sini distribution, given the actual i_o distribution and a few different assumed distributions of psi


def sini_from_assumed_psi(psi_val=None, psi_err=None):

    sini_s_means = []
    for ru in range(runs):

        sini_ss = []
        for io in science_sample['i_o']:

            if psi_err is not None:
                if psi_err == "iso":
                    psi = np.arccos(np.random.uniform(0, 1))
                else:
                    psi = np.random.rayleigh(psi_err)  # obliquity distribution for which we want to find the corresponding psi

            if psi_val is not None:
                psi = psi_val

            Omega = np.random.uniform(-np.pi * 2, np.pi * 2)  # random placement on the n_s ellipse.
            lmbda = np.arctan(np.sin(psi) * np.sin(Omega) / (np.cos(psi) * np.sin(io) + np.sin(psi) * np.cos(Omega) * np.cos(io)))

            i_s = np.arcsin(np.sin(psi) * np.sin(Omega) / np.sin(lmbda))
            sini_ss.append(np.sin(np.abs(i_s)))

        # fig, ax = plt.subplots(1, 1, figsize=(10, 10)) # checks the i_s and i_o distributions. Only run when "runs" is low
        # bins = np.linspace(0, 90, 60)
        # ax.hist(science_sample['i_o'] * 180 / np.pi, bins=bins, alpha=0.7)
        # ax.hist(np.array(i_ss)*180/np.pi, bins=bins, alpha=0.7)
        # plt.show()

        sini_s_means.append(np.mean(sini_ss))

    return sini_s_means


ax_bootstrap.hist(sini_from_assumed_psi(psi_err=5*np.pi/180), bins=bins_bootstrap, label='$\sigma \psi$ = 5$^{\circ}$', alpha=0.7)
ax_bootstrap.hist(sini_from_assumed_psi(psi_err=10*np.pi/180), bins=bins_bootstrap, label='$\sigma \psi$ = 10$^{\circ}$', alpha=0.7)
ax_bootstrap.hist(sini_from_assumed_psi(psi_err=15*np.pi/180), bins=bins_bootstrap, label='$\sigma \psi$ = 15$^{\circ}$', alpha=0.7)
ax_bootstrap.hist(sini_from_assumed_psi(psi_err=25*np.pi/180), bins=bins_bootstrap, label='$\sigma \psi$ = 25$^{\circ}$', alpha=0.7)
ax_bootstrap.hist(sini_from_assumed_psi(psi_err="iso"), bins=bins_bootstrap, label='$\psi$ iso', alpha=0.7)
ax_bootstrap.hist(sini_from_assumed_psi(psi_val=2*np.pi/180), bins=bins_bootstrap, label='$\psi$ = 2$^{\circ}$', alpha=0.7)

ax_bootstrap.legend()
ax.set_xlabel('mean sini')
plt.show()
