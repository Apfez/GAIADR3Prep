import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import GAIAfunctions as gf

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

psi_kappa = 5
i_o_kappa = 15

# science_sample = gf.get_confirmed_planet_hosts(T_eff_min, T_eff_max, logg_min, logg_max, feh_min, feh_max)
science_sample = gf.get_binaries(1000, psi_kappa, i_o_kappa)

control_sample = gf.make_control_sample(control_stars_unfiltered, science_sample)  # Takes a long time to run, so I make a new cell


################################## plot the data #############################################

fig, axs = plt.subplots(4)
fig.set_size_inches(10, 14)

# adjust space between subplots
fig.subplots_adjust(hspace=0.3)

ax = axs[0]
ax.scatter(control_sample['ra'], control_sample['dec'], c='C1', s=1)
ax.scatter(science_sample['ra'], science_sample['dec'], c='C0', s=1)
ax.set_xlabel('RA')
ax.set_ylabel('Dec')

ax = axs[1]
ax.scatter(control_sample['teff'], control_sample['logg'], c='C1', s=1)
ax.scatter(science_sample['teff'], science_sample['logg'], c='C0', s=1)
ax.set_xlabel('$T_{eff}$')
ax.set_ylabel('logg')

ax = axs[2]
ax.scatter(control_sample['teff'], control_sample['feh'], c='C1', s=1)
ax.scatter(science_sample['teff'], science_sample['feh'], c='C0', s=1)
ax.set_xlabel('$T_{eff}$')
ax.set_ylabel('[Fe/H]')

ax = axs[3]
ax.scatter(control_sample['logg'], control_sample['feh'], c='C1', s=1)
ax.scatter(science_sample['logg'], science_sample['feh'], c='C0', s=1)
ax.set_xlabel('logg')
ax.set_ylabel('[Fe/H]')

ax.legend(['%1.0f control stars'%len(control_sample), '%1.0f science stars'%len(science_sample)])
plt.show()
