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

def infer_stellar_age(df):

    # Set up the parameter dictionary.
    iso_params = {"G": (df["G"], .05),
                  "bp": (df["bp"], .05),
                  "rp": (df["rp"], .05),
                  "parallax": (df["parallax"], df["parallax_err"]),
                  "maxAV": .1}

    # Infer an age with isochrones and gyrochronology.

    try:
        sd_fn = "praesepe/{}_stardate".format(str(int(df["EPIC"]))
        iso_fn = "praesepe/{}_isochrones".format(str(int(df["EPIC"])))
        if not os.path.exists(sd_fn):
            # Set up the star object
            star = sd.Star(iso_params, df["Prot1"], 1., filename=sd_fn)

            # Run the MCMC
            sampler = star.fit(max_n=200000)

        else:
            print("failed to save file for star. File exists: ", sd_fn)

            # Now infer an age with isochrones only.

        if not os.path.exists(iso_fn):
            # Set up the star object
            star_iso = sd.Star(iso_params, None, None, filename=iso_fn)

            # Run the MCMC
            sampler = star_iso.fit(max_n=200000, iso_only=True)

        else:
            print("failed to save file for star. File exists: ", iso_fn)

    except:
        print("failed to save file for star ", str(int(df["EPIC"]))


if __name__ == "__main__":
    #  Load the Praesepe data file.
    df = pd.read_csv("data/praesepe.csv")

    list_of_dicts = []
    for i in range(len(df)):
        list_of_dicts.append(df.iloc[i].to_dict())

    p = Pool(24)
    list(p.map(infer_stellar_age, list_of_dicts))
