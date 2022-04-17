import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import GAIAfunctions as gf


def v_louden(T):
    tau = (T - 5800) / 1000

    c0 = 5.5
    c1 = 3.5
    c2 = 1.8

    v = c0 + c1 * tau + c2 * tau ** 2

    return v


# Get all stars from csv for use as control sample
control_stars_unfiltered = pd.read_csv("all_gaia_stars-result.csv")

# Add some noise to the data
control_stars_unfiltered['teff'] = control_stars_unfiltered['teff'] + np.random.uniform(-100, 100,
                                                                                        len(control_stars_unfiltered))
control_stars_unfiltered['logg'] = control_stars_unfiltered['logg'] + np.random.uniform(-0.1, 0.1,
                                                                                        len(control_stars_unfiltered))
control_stars_unfiltered['feh'] = control_stars_unfiltered['feh'] + np.random.uniform(-0.1, 0.1,
                                                                                      len(control_stars_unfiltered))

# Define parameters for the science sample
T_eff_min = 5000
T_eff_max = 10000
logg_min = 3.9
logg_max = 4.8
feh_min = -0.5
feh_max = 1

science_sample = gf.get_binaries(200)

control_sample = gf.make_control_sample(control_stars_unfiltered, science_sample)  # Takes a long time to run, so I make a new cell

################################## plot the data #############################################

# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#
# ax.hist(science_sample['inclination_o']*180/np.pi, bins=10)
# plt.show()


fig, ax = plt.subplots(1, 1, figsize=(10, 10))

sini_control = np.sin(np.arccos(np.linspace(0, 1, len(control_sample))))
vsini_control = sini_control * v_louden(control_sample['teff'])

vsini_science = np.sin(science_sample['inclination_o']) * v_louden(science_sample['teff'])

ax.scatter(control_sample['teff'], vsini_control, c='k', s=1)
ax.scatter(science_sample['teff'], vsini_science, c='r', s=1)

plt.show()

