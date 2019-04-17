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
    iso_params = {"G": (df["G"], df["G_err"]),
                  "BP": (df["bp"], df["bp_err"]),
                  "RP": (df["rp"], df["rp_err"]),
                  "parallax": (df["parallax"], df["parallax_err"]),
                  "maxAV": .1}

    # Infer an age with isochrones and gyrochronology.

    fn = "praesepe_results/{}_praesepe_stardate".format(
        str(int(df["EPIC"])).zfill(9))

    if not os.path.exists(fn):
        # Set up the star object
        star = sd.Star(iso_params, prot=df["prot"], prot_err=.05,
                        filename=fn)

        # Run the MCMC
        inits = [330, np.log10(650*1e6), 0., np.log(177), 0.035]
        star.fit(max_n=500000, inits=inits, optimize=False,
                    model="praesepe")

    else:
        print("failed to save file for star ", str(int(df["EPIC"])).zfill(9))


if __name__ == "__main__":
    print("praesepe_parallel.py")

    #  Load the Praesepe data file.
    df = pd.read_csv("data/praesepe.csv")
    assert len(df.EPIC) == len(np.unique(df.EPIC))
    df = df.iloc[:200]

    list_of_dicts = []
    for i in range(len(df)):
        list_of_dicts.append(df.iloc[i].to_dict())

    print(list_of_dicts[0])
    print(len(list_of_dicts))

    p = Pool(24)
    list(p.map(infer_stellar_age, list_of_dicts))
