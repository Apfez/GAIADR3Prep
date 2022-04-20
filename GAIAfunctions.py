import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def v_rotation(T, c0, c1, c2):
    tau = (T - 5800) / 1000

    v = c0 + c1 * tau + c2 * tau ** 2

    return v


def get_confirmed_planet_hosts(T_min, T_max, l_min, l_max, f_min, f_max):
    df = pd.read_csv('All_confirmed_planets.csv')
    df.drop_duplicates(subset="hostname", keep="first", inplace=True)

    df.rename(columns={'st_teff': 'teff'}, inplace=True)
    df.rename(columns={'st_logg': 'logg'}, inplace=True)
    df.rename(columns={'st_met': 'feh'}, inplace=True)

    mask = np.all([(df['teff'] > T_min),
                   (df['teff'] < T_max),
                   (df['logg'] > l_min),
                   (df['logg'] < l_max),
                   (df['feh'] > f_min),
                   (df['feh'] < f_max),
                   ~np.isnan(df['pl_orbper']),
                   np.isnan(df['pl_projobliq']),
                   np.isnan(df['pl_trueobliq']),
                   (df.iloc[:]['discoverymethod'] == "Transit")
                   ], axis=0)

    return df[mask]


def get_binaries(N, psi_kappa, i_o_kappa, psi_mu=0, i_o_mu=0):
    df = pd.read_csv('fake_binaries.csv')

    if N >= len(df):
        N = len(df)

    df = df[:N]

    i_os = []
    for n in range(N):

        i_o = 10000
        while i_o > 30 * np.pi / 180:  # cut-off in inclination corresponding to a potential choice we want to make to avoid degeneracy in psi
            i_o = vMF_sampler(i_o_kappa, i_o_mu)
        i_os.append(i_o)
    i_os = np.array(i_os)

    psis = vMF_sampler(psi_kappa, psi_mu, N)

    Omegas = np.random.uniform(-np.pi * 2, np.pi * 2, N)
    lmbdas = np.arctan(
        np.sin(psis) * np.sin(Omegas) / (np.cos(psis) * np.sin(i_os) + np.sin(psis) * np.cos(Omegas) * np.cos(i_os)))

    i_ss = np.arcsin(np.abs(np.sin(psis) * np.sin(Omegas) / np.sin(lmbdas)))

    df['psi'] = psis
    df['i_s'] = i_ss
    df['i_o'] = i_os
    df['i_o_err'] = np.random.uniform(2, 10, N)*np.pi/180
    df['vbroad'] = v_rotation(df['teff'], 5.5, 3.5, 2.4) * np.sin(np.array(i_ss))

    return df


def make_control_sample(cs_unfiltered, science_sample):

    # construct a control sample by selecting stars from the science sample with similar T_eff,logg,feh and perhaps magnitude.
    control_sample_ids = []
    delta_T_eff = 10
    delta_logg = 0.04
    delta_feh = 0.04
    delta_photg_mean_mag = 3

    T_eff_max = np.max(cs_unfiltered['teff'])
    T_eff_min = np.min(cs_unfiltered['teff'])
    logg_max = np.max(cs_unfiltered['logg'])
    logg_min = np.min(cs_unfiltered['logg'])
    feh_max = np.max(cs_unfiltered['feh'])
    feh_min = np.min(cs_unfiltered['feh'])

    for i in range(len(science_sample)):

        # find all stars in the science sample with similar T_eff

        mask = np.all([(cs_unfiltered['teff'] > science_sample.iloc[i]['teff'] - delta_T_eff),
                       (cs_unfiltered['teff'] < science_sample.iloc[i]['teff'] + delta_T_eff),
                       (cs_unfiltered['logg'] > science_sample.iloc[i]['logg'] - delta_logg),
                       (cs_unfiltered['logg'] < science_sample.iloc[i]['logg'] + delta_logg),
                       (cs_unfiltered['feh'] > science_sample.iloc[i]['feh'] - delta_feh),
                       (cs_unfiltered['feh'] < science_sample.iloc[i]['feh'] + delta_feh),
                       # (cs_unfiltered['phot_g_mean_mag'] > science_sample.iloc[i]['sy_gaiamag'] - delta_photg_mean_mag),
                       # (cs_unfiltered['phot_g_mean_mag'] < science_sample.iloc[i]['sy_gaiamag'] + delta_photg_mean_mag),
                       ], axis=0)

        length = int(mask.sum())

        if length > 0:
            m = np.min([length, 10])
            temp = cs_unfiltered[mask]

            temp['error'] = ((temp['teff'] - science_sample.iloc[i]['teff']) / (T_eff_max - T_eff_min) ** 2) + \
                            ((temp['logg'] - science_sample.iloc[i]['logg']) / (logg_max - logg_min)) ** 2 + \
                            ((temp['feh'] - science_sample.iloc[i]['feh']) / (feh_max - feh_min)) ** 2

            # sort temp by error
            temp = temp.sort_values(by='error')
            control_sample_ids = np.concatenate((control_sample_ids, temp.iloc[0:m].index))

    cs = cs_unfiltered[cs_unfiltered.index.isin(control_sample_ids)]

    # remove duplicate stars from the control sample
    cs.drop_duplicates(subset="source_id", keep="first", inplace=True)

    cs['vbroad'] = v_rotation(cs['teff'], 5.5, 3.5, 2.4) * np.sin(np.arccos(np.random.uniform(0, 1, len(cs))))

    return cs


def p_fisher(psi, mu, kappa):

    if kappa == 0:  # as to not divide by zero
        return np.sin(psi)
    elif kappa < 100:
        return kappa / (np.exp(kappa) - np.exp(-kappa)) * np.exp(kappa * np.cos(psi - mu)) * np.sin(psi)
    else:  # to avoid overflow
        sigma = kappa ** (-0.5)
        return psi / (sigma ** 2) * np.exp(-psi ** 2 / (2 * sigma ** 2))


def vMF_sampler(kappa, mu, size=1):

    if kappa >= 100:
        sigma = kappa ** (-0.5)
        samples = np.random.rayleigh(sigma, size)
    else:
        samples = []
        scaling = np.max(p_fisher(np.linspace(0, np.pi, 10000), mu, kappa))

        while len(samples) < size:  # uses a kind of accept-reject method to sample from a von Mises-Fisher distribution

            psi_rand = np.random.uniform(0, np.pi)
            height = np.random.uniform(0, 1)

            if height < p_fisher(psi_rand, mu, kappa) / scaling:
                samples.append(psi_rand)

    if size == 1:
        return samples[0]

    return np.array(samples)
