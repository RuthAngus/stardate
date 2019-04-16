import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm, trange

import read_mist_models
from isochrones.mist import MIST_Isochrone
from isochrones import StarModel

from stardate.lhf import gyro_model_rossby, gyro_model, sigma, calc_rossby_number

bands = ["B", "V", "J", "H", "K", "BP", "RP", "G"]
mist = MIST_Isochrone(bands)
mist.initialize()

def color_err(c):
    c_err = np.zeros(len(c))
    bright = c < 13
    medium = (13 < c) * (c < 17)
    faint = 17 <= c
    c_err[bright] = np.ones(len(c_err[bright]))*.002
    c_err[medium] = np.ones(len(c_err[medium]))*.01
    c_err[faint] = np.ones(len(c_err[faint]))*.2
    return c_err


def photometric_noise(G, bp, rp):
    G_err = np.zeros(len(G))
    bright = G < 13
    medium = (13 < G) * (G < 17)
    faint = 17 <= G
    G_err[bright] = np.ones(len(G_err[bright]))*.0003
    G_err[medium] = np.ones(len(G_err[medium]))*.002
    G_err[faint] = np.ones(len(G_err[faint]))*.01

    bp_err = color_err(bp)
    rp_err = color_err(rp)

    parallax_err = np.zeros(len(G))
    bright = G < 15
    medium = (15 < G) * (G < 17)
    faint = (17 < G) * (G < 20)
    ultra_faint = 21 <= G
    parallax_err[bright] = .03  # milliarcseconds
    parallax_err[medium] = .1
    parallax_err[faint] = .7
    parallax_err[ultra_faint] = 2

    return G_err, bp_err, rp_err, parallax_err


def log_error_abs(x, dx):
    return dx / x*np.log(10)

def log_error(rel_errx):
    return rel_errx / np.log(10)


def generate_df(N=1000, with_noise=False):
    """
    Simulate stellar properties from an distribution of EEPss, ages,
    metallicities, distances and extinctions.
    """

    np.random.seed(42)
    eep = np.random.uniform(195, 480, size=N)
    age = np.log10(np.random.uniform(.5, 14, size=N)*1e9)
    feh = np.random.uniform(-.2, .2, size=N)
    mass = mist.interp_value([eep, age, feh], ["mass"])
    d_kpc = np.random.uniform(.01, 1, size=N)
    d_pc = d_kpc*1e3
    av = np.random.uniform(0, .1, size=N)

    # Save as a pandas data frame
    df = pd.DataFrame(dict({"eep": eep, "age": age, "feh": feh, "d_kpc": d_kpc, "Av": av}))

    logg, teff, feh_interp, B, V, J, H, K, BP, RP, G, logL = [np.zeros(N) for i in range(12)]
    print("Calculating stellar parameters...")
    for i in trange(N):
        teff[i], logg[i], feh_interp[i], bands = mist.interp_mag(
            [eep[i], age[i], feh[i], d_pc[i], av[i]],
            ["B", "V", "J", "H", "K", "BP", "RP", "G"])
        B[i], V[i], J[i], H[i], K[i], BP[i], RP[i], G[i] = bands
        logL[i] = mist.interp_value([eep[i], age[i], feh[i]], ["logL"])[0]
    parallax = 1./d_kpc

    G_err, bp_err, rp_err, parallax_err = photometric_noise(G, BP, RP)
    df["BP_err"] = bp_err
    df["RP_err"] = rp_err
    df["G_err"] = G_err
    df["parallax_err"] = parallax_err

    df["B_err"] = np.ones_like(N)*.01
    df["V_err"] = np.ones_like(N)*.01
    df["J_err"] = np.ones_like(N)*.01
    df["H_err"] = np.ones_like(N)*.01
    df["K_err"] = np.ones_like(N)*.01
    df["teff_err"] = np.ones_like(N)*25
    df["logg_err"] = np.ones_like(N)*.01
    df["feh_err"] = np.ones_like(N)*.01

    if with_noise:
        np.random.seed(42)
        B += np.random.randn(N)*df["B_err"]
        V += np.random.randn(N)*df["V_err"]
        J += np.random.randn(N)*df["J_err"]
        K += np.random.randn(N)*df["H_err"]
        H += np.random.randn(N)*df["K_err"]
        BP += np.random.randn(N)*df["BP_err"]
        RP += np.random.randn(N)*df["RP_err"]
        G += np.random.randn(N)*df["G_err"]
        teff += np.random.randn(N)*df["teff_err"]
        logg += np.random.randn(N)*df["logg_err"]
        feh += np.random.randn(N)*df["feh_err"]
        parallax += np.random.randn(N)*df["parallax_err"]


    df["BV"], df["B"], df["V"], df["jmag"], df["hmag"], df["kmag"] = B - V, B, V, J, H, K
    df["BP"], df["RP"], df["G"] = BP, RP, G
    df["logg"], df["teff"], df["logL"], df["parallax"] = logg, teff, logL, parallax
    df["mass"] = mist.mass(df.eep, df.age, df.feh)

    # Calculate periods. NaNs will appear for stars with B-V < 0.45
    np.random.seed(42)
    log_prot, prot_err, log_prot_praesepe, prot_err_praesepe = [np.empty(N) for i in range(4)]
    log_prot_scatter, log_prot_scatter_praesepe = [np.empty(N) for i in range(2)]
    period_uncertainty = .1  #  relative uncertainty of 10%
    print("Calculating rotation periods...")
    for i in trange(N):
        params = [eep[i], age[i], feh[i], d_pc[i], av[i]]
        lpa, siga = gyro_model_rossby(params, rossby=True, Ro_cutoff=2, model="angus15")
        log_prot[i] = lpa
        lpa, sigp = gyro_model_rossby(params, rossby=True, Ro_cutoff=2, model="praesepe")
        log_prot_praesepe[i] = lpa
#         prot_err[i] = (10**log_prot[i])*period_uncertainty
#         prot_err_praesepe[i] = (10**log_prot_praesepe[i])*period_uncertainty

        # Add variance to these.
        scattera = np.random.randn(1)*siga
        log_prot_scatter[i] = log_prot[i] + scattera
        scatterp = np.random.randn(1)*sigp
        log_prot_scatter_praesepe[i] = log_prot_praesepe[i] + scatterp

#     # Add variance to these.
#     sig_praesepe = np.zeros(N)
#     for i in trange(N):
#         sig_praesepe[i] = sigma(df.BP.values[i] - df.RP.values[i], df.eep.values[i], model="praesepe")
#     scatter = np.random.randn(len(log_prot_praesepe))*sig_praesepe
#     log_prot_scatter_praesepe = log_prot_praesepe + scatter

    noise = 0
    if with_noise:
        noise = np.random.randn(N) * log_error(period_uncertainty)

    df["prot"] = 10**(log_prot_scatter + noise)
    df["prot_praesepe"] = 10**(log_prot_scatter_praesepe + noise)
    df["prot_clean"] = 10**log_prot
    df["prot_clean_prasepe"] = 10**log_prot_praesepe
    df["prot_err"] = df["prot"]*period_uncertainty
    df["prot_err_praesepe"] = df["prot_praesepe"]*period_uncertainty
    df["Ro"] = calc_rossby_number(df.prot, df.mass)
    df["Ro_praesepe"] = calc_rossby_number(df.prot_praesepe, df.mass)

    # Cut unphysical masses
    finite_mask = np.isfinite(df.mass.values)
    finite_df = df.iloc[finite_mask]
    print(len(df.mass.values), "stars originally, ", len(df.mass.values[finite_mask]), "after cuts")

    finite_df["ID"] = range(len(finite_df))

    return finite_df
