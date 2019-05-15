import numpy as np
import stardate as sd
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt
from stardate.lhf import lnprob
from stardate import load_samples, read_samples
from isochrones import StarModel, get_ichrone
bands = ["B", "V", "J", "H", "K", "BP", "RP"]
mist = get_ichrone("mist", bands=bands)
import corner


def test_sim():
    full_df = pd.read_csv("../../paper/code/data/simulated_data.csv")

    i = 15
    df = full_df.iloc[i]

    teff_err, logg_err, feh_err = 25, .05, .05

    iso_params = {"teff": (df["teff"], teff_err),
                 "logg": (df["logg"], logg_err),
                 "feh": (df["feh"], feh_err),
                 "J": (df["jmag"], df["J_err"]),
                 "H": (df["hmag"], df["H_err"]),
                 "K": (df["kmag"], df["K_err"]),
                 "B": (df["B"], df["B_err"]),
                 "V": (df["B"], df["V_err"]),
                 "G": (df["G"], df["G_err"]),
                 "BP": (df["BP"], df["BP_err"]),
                 "RP": (df["RP"], df["RP_err"]),
                 "parallax": (df["parallax"], df["parallax_err"]),
                 "maxAV": .1}
    print(df["parallax"], df["parallax_err"])

    mod = StarModel(mist, **iso_params)
    args = [mod, df["prot_praesepe"], df["prot_err_praesepe"], False, False,
            True, "praesepe"]
    inits=[329.58, 9.5596, -.0478, np.log(260), .0045]
    print(lnprob(inits, *args))
    assert 0

    # lnparams = [df.eep, df.age, df.feh, np.log(df.d_kpc*1e3), df.Av]
    # print(lnprob(lnparams, *args))
    filename = "{}_M_test".format(str(int(df["ID"])).zfill(4))
    star = sd.Star(iso_params, prot=df["prot"], prot_err=.01,
                   filename=filename)
    star.fit(max_n=20000, model="praesepe")


def test_praesepe():
    df = pd.read_csv("data/praesepe.csv")
    df = df.iloc[4]

    iso_params = {"G": (df["G"], df["G_err"]),
                  "BP": (df["bp"], df["bp_err"]),
                  "RP": (df["rp"], df["rp_err"]),
                  "parallax": (df["parallax"], df["parallax_err"]),
                  "maxAV": .1}
    print(iso_params)
    print(df["prot"])

    fn = "{}_praesepe_test".format(str(int(df["EPIC"])).zfill(9))
    star = sd.Star(iso_params, prot=df["prot"], prot_err=.1,
                   # color=df["bp"]-df["rp"], mass=1., filename=fn)
                   color=.65, mass=1., filename=fn)

    inits = [330, np.log10(650*1e6), 0., np.log(177), 0.035]
    sampler = star.fit(max_n=20000, inits=inits, gyro_only=False,
                       optimize=False, model="angus15", rossby=True)


def test_on_sun():
    iso_params = {"teff": (5778, 50),
                  "logg": (4.44, .1),
                  "feh": (0., .1),
                  "parallax": (1, .1),
                  "maxAV": .1}
    fn = "star_test"
    star = sd.Star(iso_params, prot=26., prot_err=26*.01, Av=0., Av_err=.001,
                   savedir=".", filename=fn)
    star.fit(max_n=1000)
    samples = star.samples

    flatsamples, _ = load_samples("{0}.h5".format(fn))
    assert np.shape(flatsamples[:, :-1]) == np.shape(samples)
    results = read_samples(flatsamples)

    # labels = ["EEP", "Age", "Fe/H", "Distance", "Av", "lnprob"]
    # truths = [355, np.log10(4.56*1e9), 0., 1000, 0., None]
    # corner.corner(flatsamples, labels=labels, truths=truths);
    # plt.savefig("sun_test_corner")

    # Assert the results are within 1 sigma of the Sun.
    assert (float(results.age_med_gyr) - float(results.age_errm) < 4.56), \
        "Solar age too high"
    assert (4.56 < float(results.age_med_gyr) + float(results.age_errp)), \
        "Solar age too low"
    assert (float(results.EEP_med) - float(results.EEP_errm) < 355), \
        "Solar EEP too high"
    assert (255 < float(results.EEP_med) + float(results.EEP_errp)), \
        "Solar EEP too low"


def test_hyades():
    i = 1
    df = pd.read_csv("../../paper/code/data/hyades_single.csv")

    iso_params = {"G": (df.G[i], df.G_err[i]),
                "BP": (df.bp[i], df.bp_err[i]),
                "RP": (df.rp[i], df.rp_err[i]),
                "parallax": (df.parallax[i], df.parallax_err[i]),
                "maxAV": .1}

    inits = [330, np.log10(650*1e6), .07, np.log(45), 0.]
    star = sd.Star(iso_params, prot=df.prot[i], prot_err=df.prot[i]*.1,
                   filename="hyades_test")
    print(df.prot[i], "prot")

    mod = StarModel(mist, **iso_params)
    args = [mod, df.prot.values[i], df.prot.values[i]*.1, False, True,
            "praesepe"]
    print(lnprob(inits, *args))
    assert 0


if __name__ == "__main__":
    # test_sim()
    # test_praesepe()
    test_on_sun()
    # test_hyades()
