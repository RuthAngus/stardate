#!/usr/bin/python3

import os
import sys
import numpy as np
import pandas as pd
from isochrones import get_ichrone
import stardate as sd
from multiprocessing import Pool
from isochrones.mist import MIST_Isochrone
from isochrones import SingleStarModel
import emcee
bands = ["B", "V", "J", "H", "K", "BP", "RP", "G"]
mist = get_ichrone("mist", bands=bands)

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
sys.path.append(os.getcwd())


def iso_lnprob(params, *args):
    mod, iso_params = args
    like, prior = mod.lnlike(params), mod.lnprior(params)
    prob = like + prior
    if not np.isfinite(prob):
        prob = -np.inf
    return prob


def infer_stellar_age(df):

    iso_params = {
                  "G": (df["G"], df["G_err"]),
                  "BP": (df["bp"], df["bp_err"]),
                  "RP": (df["rp"], df["rp_err"]),
                  "g": (df["g_final"], df["g_final_err"]),
                  "r": (df["r_final"], df["r_final_err"]),
                  "J": (df["Jmag"], df["e_Jmag"]),
                  "H": (df["Hmag"], df["e_Hmag"]),
                  "K": (df["Kmag"], df["e_Kmag"]),
                  "teff": (df["teff_vansaders"], df["teff_err"]),
                  "feh": (df["feh_vansaders"], df["feh_err"]),
                  "nu_max": (df["numax"], df["e_numax"]),
                  "delta_nu": (df["Dnu"], df["e_Dnu"]),
                  "parallax": (df["parallax"], df["parallax_error"])}

    mist = get_ichrone('mist')
    age_init = np.log10(df["AMP_age"]*1e9)
    eep = mist.get_eep(df["AMP_mass"], age_init, df["feh_vansaders"],
                       accurate=True)

    mod = SingleStarModel(mist, **iso_params)  # StarModel isochrones obj
    params = [354, np.log10(4.56*1e9), 0., 1000, 0.]
    args = [mod, iso_params]

    nwalkers, ndim, nsteps = 50, 5, 10000
    inits = [eep, age_init, df["feh_vansaders"], 1./df["parallax"]*1e3,
             df["Av"]]
    p0 = [np.random.randn(ndim)*1e-4 + inits for j in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, iso_lnprob, args=args);
    sampler.run_mcmc(p0, nsteps);
    samples = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))

    sample_df = pd.DataFrame({"eep_samples": samples[:, 0],
                              "age_samples": samples[:, 1],
                              "feh_samples": samples[:, 2],
                              "distance_samples": samples[:, 3],
                              "Av_samples": samples[:, 4]})
    sample_df.to_hdf("{}_iso_test.h5".format(int(df["kepid"])), key="samps",
                     mode='w')


if __name__ == "__main__":
    df = pd.read_csv("data/astero_gaia.csv")

    list_of_dicts = []
    for i in range(len(df)):
        list_of_dicts.append(df.iloc[i].to_dict())

    p = Pool(21)
    list(p.map(infer_stellar_age, list_of_dicts))
