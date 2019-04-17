#!/usr/bin/python3

import os
import sys
import numpy as np
import pandas as pd
import h5py
import tqdm
import emcee

import stardate as sd
from stardate.lhf import lnprob

from multiprocessing import Pool

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
sys.path.append(os.getcwd())

# def infer_stellar_age(row):
#     df = row[1]
def infer_stellar_age(df):

    # Small observational uncertainties are needed (even though the stars
    # weren't simulated with any) in order to get a good fit.
    teff_err = 25  # Kelvin
    logg_err = .05  # dex
    feh_err = .05  # dex
    jmag_err = .01 # mags
    hmag_err = .01  # mags
    kmag_err = .01  # mags
    B_err, V_err, bp_err, rp_err = .01, .01, .01, .01
    parallax_err = .05  # milliarcseconds
    prot_err = 1  # Days
    BV_err = .01  # mags

    #  Infer ages of the simulated stars.

    # Set up the parameter dictionary.
    iso_params = {"teff": (df["teff"], teff_err),
                  "logg": (df["logg"], logg_err),
                  "feh": (df["feh"], feh_err),
                  "J": (df["jmag"], jmag_err),
                  "H": (df["hmag"], hmag_err),
                  "K": (df["kmag"], kmag_err),
                  "B": (df["B"], B_err),
                  "V": (df["V"], V_err),
                  "G": (df["G"], bp_err),
                  "BP": (df["BP"], bp_err),
                  "RP": (df["RP"], rp_err),
                  "parallax": (df["parallax"], parallax_err),
                  "maxAV": .1}

    # Infer an age with isochrones and gyrochronology.

    try:
        sd_fn = "simulation_results/{}_stardate".format(
            str(int(df["ID"])).zfill(4))

        if not os.path.exists(sd_fn):
            # Set up the star object
            star = sd.Star(iso_params, prot=df["prot"], prot_err=.01,
                           filename=sd_fn)

            # Run the MCMC
            sampler = star.fit(max_n=500000)

        else:
            print("failed to save file for star. File exists: ", sd_fn)

    except:
        print("failed to save file for star ", str(int(df["ID"])).zfill(4))


if __name__ == "__main__":
    #  Load the simulated data file.
    df = pd.read_csv("data/simulated_data.csv")
    assert len(df.ID) == len(np.unique(df.ID))
    df = df.iloc[:200]

    list_of_dicts = []
    for i in range(len(df)):
        list_of_dicts.append(df.iloc[i].to_dict())

    print(list_of_dicts[0])
    print(len(list_of_dicts))

    p = Pool(24)
    list(p.map(infer_stellar_age, list_of_dicts))
