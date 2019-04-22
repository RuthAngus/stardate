import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm, trange
import emcee

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
    """
    Calculates the noise on Gaia G, bp, rp and parallax.

    Args:
        G (array): Gaia G band magnitude.
        bp (array): Gaia G_BP band magnitude.
        rp (array): Gaia G_RP band magnitude.

    Returns:
        G (array): Uncertainty on Gaia G band magnitude.
        bp (array): Uncertainty on Gaia G_BP band magnitude.
        rp (array): Uncertainty on Gaia G_RP band magnitude.
        parallax_err (array): Photometry-based uncertainty on Gaia parallax.

    """
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


def percentiles_from_samps(samps):
    med = np.median(samps)
    std = np.std(samps)
    upper = np.percentile(samps, 84)
    lower = np.percentile(samps, 16)
    errp = upper - med
    errm = med -lower
    return med, errp, errm, std


def read_files(name, ids, dirname=".", zf=4, burnin=100):
    """
    Read h5 results files.

    Args:
        name (int): The non star-id part of the string.
        ids (list): list of star ids.
        dirname (str): Directory path.
        burnin (Optional[int]): Number of burn in samples to discard.

    Returns:
        age_samps (list): list of arrays.
        meds (array): median age values.
        errp (array): upper age uncertainties.
        errm (array): lower age uncertainties.
        std (array): Age uncertainties.
        inds (array): Indices of ids that had files available.
    """
    meds, age_samps, errp, errm, std, inds = [], [], [], [], [], []

    for i, ID in enumerate(ids):
        try:
            fname = "{0}/{1}_{2}.h5".format(dirname, str(int(ID)).zfill(zf),
                                            name)
            reader = emcee.backends.HDFBackend(fname)
            samples = reader.get_chain()

            if np.shape(samples)[0] > 10:
                nsteps, nwalkers, ndim = np.shape(samples)
                samps = np.reshape(samples, (nsteps*nwalkers, ndim))

                a, ap, am, _std = percentiles_from_samps(
                    (10**samps[burnin:, 1])*1e-9)
                age_samps.append((10**samps[burnin:, 1])*1e-9)
                meds.append(a)
                errp.append(ap)
                errm.append(am)
                std.append(_std)
                inds.append(i)
            else:
                print("too few samples")

        except:
            pass

    return age_samps, np.array(meds), np.array(errp), np.array(errm), \
        np.array(std), np.array(inds)


def getDust(G, bp, rp, ebv, maxnit=100):
    """
    Compute the Gaia extinctions assuming relations from Babusieux.

    Author: Sergey Koposov skoposov@cmu.edu

    Args:
        G (float): Gaia G mag.
        bp (float): Gaia BP mag.
        rp (float): Gaia RP mag.
        ebv (float): E(B-V), extinction in B-V.
        maxnit (int): number of iterations

    Returns:
        Extinction in G,bp, rp

    """
    c1, c2, c3, c4, c5, c6, c7 = [0.9761, -0.1704,
                                  0.0086, 0.0011, -0.0438, 0.0013, 0.0099]
    d1, d2, d3, d4, d5, d6, d7 = [
        1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043]
    e1, e2, e3, e4, e5, e6, e7 = [
        0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006]
    A0 = 3.1*ebv
    P1 = np.poly1d([c1, c2, c3, c4][::-1])

    def F1(bprp): return np.poly1d(
        [c1, c2, c3, c4][::-1])(bprp)+c5*A0+c6*A0**2+c7*bprp*A0

    def F2(bprp): return np.poly1d(
        [d1, d2, d3, d4][::-1])(bprp)+d5*A0+d6*A0**2+d7*bprp*A0

    def F3(bprp): return np.poly1d(
        [e1, e2, e3, e4][::-1])(bprp)+e5*A0+e6*A0**2+e7*bprp*A0
    xind = np.isfinite(bp+rp+G)
    curbp = bp-rp
    for i in range(maxnit):
        AG = F1(curbp)*A0
        Abp = F2(curbp)*A0
        Arp = F3(curbp)*A0
        curbp1 = bp-rp-Abp+Arp

        delta = np.abs(curbp1-curbp)[xind]
        curbp = curbp1

    AG = F1(curbp)*A0
    Abp = F2(curbp)*A0
    Arp = F3(curbp)*A0
    return AG, Abp, Arp


# Add Angus (2015) model
def bv_2_bprp(bv):
    """
    Numbers from https://arxiv.org/pdf/1008.0815.pdf
    """
    a, b, c, d = .0981, 1.4290, -.0269, .0061  # sigma = .43
    return a + b*bv + c*bv**2 + d*bv**3


def bprp_2_bv(bprp):
    """
    Try to find the analytic version of this, please!
    """
    bv_iter = np.linspace(0., 2., 10000)
    bprp_pred = [bv_2_bprp(bv) for bv in bv_iter]
    diffs = bprp - np.array(bprp_pred)
    return bv_iter[np.argmin(diffs**2)]
