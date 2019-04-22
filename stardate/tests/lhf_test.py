import numpy as np
import pandas as pd
from stardate.lhf import lnprob, calc_bv, gyro_model
from stardate.lhf import convective_overturn_time, gyro_model_praesepe
from stardate.lhf import gyro_model_rossby, sigma, age_model
from tqdm import trange

# from isochrones.mist import MIST_Isochrone
# mist = MIST_Isochrone(bands)
from isochrones import StarModel, get_ichrone
bands = ["B", "V", "J", "H", "K", "BP", "RP"]
mist = get_ichrone("mist", bands=bands)


def good_vs_bad(good_lnprob, good_lnparams, args, nsamps):
    """
    Compare the likelihood of nsamps random parameter values against the
    likelihood of the true values.
    The true values should be higher.
    """
    bad_lnparams = good_lnparams*1
    bad_lnparams[0] = 10 + good_lnparams[0]  # eep
    bad_lnparams[1] = .5 + good_lnparams[1]  # age
    bad_lnparams[2] = .05 + good_lnparams[2]  # feh
    bad_lnparams[3] = .1 + good_lnparams[3]  # dist
    bad_lnprob = lnprob(bad_lnparams, *args)
    assert bad_lnprob[0] < good_lnprob[0], \
        "True parameters values must give a higher likelihood than" \
        " wrong values"


def test_lnprob_higher_likelihood_sun():
    """
    Make sure the likelihood goes down when the parameters are a worse fit.
    Test on Solar values.
    """

    iso_params = {"teff": (5777, 10),
                  "logg": (4.44, .05),
                  "feh": (0., .001),
                  "parallax": (1., .01),  # milliarcseconds
                  "B": (15.48, 0.02)}

    # Set up the StarModel isochrones object.
    mod = StarModel(mist, **iso_params)
    # the lnprob arguments]
    args = [mod, 26., 1., False, False, True, "angus15"]

    good_lnparams = [346, np.log10(4.56*1e9), 0., np.log(1000), 0.]
    good_lnprob = lnprob(good_lnparams, *args)
    good_vs_bad(good_lnprob, good_lnparams, args, 10)


def test_lnprob_higher_likelihood_real():
    """
    The same test as above but for simulated data.
    """
    df = pd.read_csv("../../paper/code/data/simulated_data.csv")
    teff_err = 25  # Kelvin
    logg_err = .05  # dex
    feh_err = .05  # dex
    jmag_err = .01 # mags
    hmag_err = .01  # mags
    kmag_err = .01  # mags
    parallax_err = .05  # milliarcseconds
    prot_err = 1  # Days
    BV_err = .01  # mags

    i = 0
    iso_params = pd.DataFrame(dict({"teff": (df.teff[i], 25),
                                    "logg": (df.logg[i], .05),
                                    "feh": (df.feh[i], .05),
                                    "jmag": (df.jmag[i], .01),
                                    "hmag": (df.hmag[i], .01),
                                    "kmag": (df.kmag[i], .01),
                                    "parallax": (df.parallax[i], .05)}))

    # Set up the StarModel isochrones object.
    mod = StarModel(mist, **iso_params)
    # lnprob arguments
    args = [mod, df.prot[i], 1, False, False, True, "angus15"]
    good_lnparams = [df.eep.values[i], df.age.values[i], df.feh.values[i],
                     np.log(df.d_kpc.values[i]*1e3), df.Av.values[i]]
    good_lnprob = lnprob(good_lnparams, *args)
    good_vs_bad(good_lnprob, good_lnparams, args, 10)


def test_for_nans():
    """
    Something is causing the lhf to return NaN. Get to the bottom of it!
    """
    df = pd.read_csv("../paper/code/data/simulated_data.csv")
    for i in range(len(df)):
        print(i, "of", len(df))
        iso_params = pd.DataFrame(dict({"teff": (df.teff[i], 10),
                                "logg": (df.logg[i], .05),
                                "feh": (df.feh[i], .001),
                                "parallax": (df.parallax[i], .01)}))  # mas

        mod = StarModel(mist, **iso_params)

        N = 10000
        eeps = np.random.uniform(-100, 2000, N)
        lnages = np.random.uniform(0, 11, N)
        fehs = np.random.uniform(-5, 5, N)
        Ds = np.log(np.random.uniform(0, 10000, N))
        Avs = np.random.uniform(-.2, 1.2, N)
        # periods = 10**np.random.uniform(-1, 3, N)
        probs, priors = [np.empty(N) for i in range(2)]

        for j in trange(N):
            lnparams = [eeps[j], lnages[j], fehs[j], Ds[j], Avs[j]]
            args = [mod, df.prot[i], 1., False, False, True,
                    "angus15"]
            probs[j] = lnprob(lnparams, *args)[0]
            priors[j] = lnprob(lnparams, *args)[1]

        print(len(probs), len(probs[np.isnan(probs)]))
        assert sum(np.isnan(probs)) == 0


def test_likelihood_rotation_giant():
    """
    Make sure that the lhf can cope with zeros, NaNs and None values for the
    rotation period.
    Also, check that there is a drop in likelihood at the eep = 454 boundary.
    """
    iso_params = pd.DataFrame(dict({"teff": (5777, 10),
                                "logg": (4.44, .05),
                                "feh": (0., .001),
                                "parallax": (1., .01)}))  # mas
    # Set up the StarModel isochrones object.
    mod = StarModel(mist, **iso_params)
    lnparams = [355, np.log10(4.56*1e9), 0., np.log(1000), 0.]

    args = [mod, None, None, False, False, True, "angus15"]  # the lnprob arguments]
    none_lnprob = lnprob(lnparams, *args)

    args = [mod, np.nan, np.nan, False, False, True, "angus15"]  # the lnprob arguments]
    nan_lnprob = lnprob(lnparams, *args)

    args = [mod, 0., 0., False, False, True, "angus15"]  # the lnprob arguments]
    zero_lnprob = lnprob(lnparams, *args)

    args = [mod, 26., 1., True, False, True, "angus15"]  # the lnprob arguments]
    iso_lnprob = lnprob(lnparams, *args)

    # check that gyro is switched off for all of these.
    none_lnprob == nan_lnprob
    nan_lnprob == zero_lnprob

    # Likelihood should be greater for dwarfs because gyro lnlike is a broad
    # Gaussian for giants.
    giant_params = [455, np.log10(4.56*1e9), 0., np.log(1000), 0.]
    dwarf_params = [453, np.log10(4.56*1e9), 0., np.log(1000), 0.]
    args = [mod, 26., 1., False, False, True, "angus15"]
    giant_lnprob = lnprob(giant_params, *args)
    dwarf_lnprob = lnprob(dwarf_params, *args)
    assert giant_lnprob[0] < dwarf_lnprob[0]

    # Likelihood should be greater for cool stars because gyro lnlike is a
    # broad Gaussian for giants.
    heep, hage, hfeh = 405, np.log10(2.295*1e9), 0.
    ceep, cage, cfeh = 355, np.log10(4.56*1e9), 0.
    hot_params = [heep, hage, hfeh, np.log(1000), 0.]
    cool_params = [ceep, cage, cfeh, np.log(1000), 0.]
    cool_prot = gyro_model_rossby(cool_params)
    hot_prot = gyro_model_rossby(hot_params)
    cool_args = [mod, cool_prot, 1., False, False, True, "angus15"]
    hot_args = [mod, hot_prot, 1., False, False, True, "angus15"]

    hot_lnprob = lnprob(hot_params, *args)
    cool_lnprob = lnprob(cool_params, *args)
    assert hot_lnprob[0] < cool_lnprob[0], "cool star likelihood should be" \
        " higher than hot star likelihood"


def test_calc_bv():
    sun = [355, np.log10(4.56*1e9), 0., np.log(1000), 0.]
    bv_sun = calc_bv(sun)
    assert .6 < bv_sun, "Are you sure you're on the isochrones eep branch?"
    assert bv_sun < .7


def test_convective_overturn_timescale():
    tau = convective_overturn_time(355, np.log10(4.56*1e9), 0.)
    tau = convective_overturn_time(1)
    solarRo = 2.16  # van Saders 2016 (2.08 in van Saders 2018)
    solarProt = 26
    solartau = solarProt / solarRo

    assert 10 < solartau and solartau < 15

    # print(26/convective_overturn_time(1.))  # 1.8
    # print(31/convective_overturn_time(1.))  # 2.14
    # print(32/convective_overturn_time(1.))  # 2.21

    # convective overturn time should go down with mass
    assert convective_overturn_time(.8) > convective_overturn_time(1.2), \
        "convective overturn time should go down with mass"


def test_gyro_model():
    """
    Check that the gyrochronology model in the lhf gives the Solar rotation
    period for Solar age and b-v.
    """
    prot = gyro_model(np.log10(4.56*1e9), .65)
    assert 24.5 < prot < 27.5
    prot = gyro_model(np.log10(4.56*1e9, .2))
    assert prot == 0


def test_praesepe_gyro_model():
    prot = gyro_model_praesepe(np.log10(4.56*1e9), .82)
    print(10**prot, "praesepe_gyro_model")
    assert 24.5 < 10**prot < 27.5


def test_gyro_model_rossby():
    """
    Make sure that the rossby model gives Solar values for the Sun.
    Make sure it gives a maximum rotation period of pmax.
    """
    age = np.log10(4.56*1e9)
    sun = [355, age, 0., np.log(1000), 0.]

    # iso_params = {"teff": (5777, 10),
    #               "logg": (4.44, .05),
    #               "feh": (0., .001),
    #               "parallax": (1., .01),  # milliarcseconds
    #               "B": (15.48, 0.02)}

    # # Set up the StarModel isochrones object.
    # mod = StarModel(mist, **iso_params)
    # # the lnprob arguments]
    # args = [mod, 26., 1., False, True, "praesepe"]
    # print(lnprob(sun, *args))
    # assert 0

    prot_sun = 10**gyro_model_rossby(sun, model="praesepe")[0]
    assert 21 < prot_sun
    assert prot_sun < 27

    prot_sun_p = 10**gyro_model_rossby(
        [355, np.log10(4.56*1e9), 0., 1000, 0.], model="praesepe")[0]
    print(prot_sun_p, "gyro_model_rossby")
    assert 21 < prot_sun_p
    assert prot_sun_p < 27


def test_sigma():
    assert sigma(355, 9, 0., .3) > .49  # Low BV (hot star) high variance
    assert sigma(355, 9, 0, 2) > .49  # high BV (cool star) high variance
    assert sigma(355, 9, 0, .65) < .01  # low variance for FGK dwarfs
    assert sigma(500, 9, 0, .65) > .49  # high variance for giants
    # assert sigma(300, 7, 0, .65) > .49  # high variance for young stars
    # assert sigma(300, 10.1, 0, .65) > .49  # high variance for old stars
    # assert sigma(355, 9, -4., .65) > .49  # high variance for metal poor
    # assert sigma(355, 9, 4., .65) > .49  # high variance for metal rich


def test_age_model():
    assert age_model(np.log10(26), 10**(-.3)) == 10.14
    assert 4.5 < (10**age_model(np.log10(26), .82))*1e-9
    assert (10**age_model(np.log10(26), .82))*1e-9 < 4.7


def test_gyro_model_praesepe():
    print(10**gyro_model_praesepe(np.log10(4.56*1e9), .77), "gyro_model_praesepe")
    assert 25 < 10**gyro_model_praesepe(np.log10(4.56*1e9), .77)
    assert 10**gyro_model_praesepe(np.log10(4.56*1e9), .77) < 27
    assert 10**gyro_model_praesepe(np.log10(4.56*1e9), 0.4) == 10**.56


def test_praesepe_angus_model():
    df = pd.read_csv("../../paper/code/data/praesepe.csv")
    df = df.iloc[0]

    iso_params = {"G": (df["G"], df["G_err"]),
                  "BP": (df["bp"], df["bp_err"]),
                  "RP": (df["rp"], df["rp_err"]),
                  "parallax": (df["parallax"], df["parallax_err"]),
                  "maxAV": .1}

    inits = [330, np.log10(650*1e6), 0., np.log(177), 0.035]
    mod = StarModel(mist, **iso_params)
    args = [mod, df["prot"], df["prot"]*.01, False, False,
            True, "angus15"]
    prob, prior = lnprob(inits, *args)
    assert np.isfinite(prob)
    assert np.isfinite(prior)


def test_on_hot_star():
    df = pd.read_csv("../../paper/code/data/simulated_data_noisy.csv")
    i = 21
    iso_params = dict({"teff": (df.teff.values[i], df.teff_err.values[i]),
                      "logg": (df.logg.values[i], df.logg_err.values[i]),
                      "feh": (df.feh.values[i], df.feh_err.values[i]),
                      "J": (df.jmag.values[i], df.J_err.values[i]),
                      "H": (df.hmag.values[i], df.H_err.values[i]),
                      "K": (df.kmag.values[i], df.K_err.values[i]),
                      "B": (df.B.values[i], df.B_err.values[i]),
                      "V": (df.V.values[i], df.V_err.values[i]),
                      "G": (df.G.values[i], df.G_err.values[i]),
                      "BP": (df.BP.values[i], df.BP_err.values[i]),
                      "RP": (df.RP.values[i], df.RP_err.values[i]),
                      "parallax": (df.parallax.values[i],
                                   df.parallax_err.values[i]),
                      "maxAV": .1})

    lnparams = [329.58, np.log10(650*1e6), 0., np.log(177), .035]
    mod = StarModel(mist, **iso_params)
    args = [mod, df.prot.values[i], df.prot_err.values[i], False, False, True,
            "angus15"]
    prob, prior = lnprob(lnparams, *args)
    print(prob, prior)
    assert np.isfinite(prob)
    assert np.isfinite(prior)


if __name__ == "__main__":

    print("\n Testing on hot star")
    test_on_hot_star()

    print("\n Testing gyro model Angus15...")
    test_praesepe_angus_model()

    print("\nTesting gyro model Rossby...")
    test_gyro_model_rossby()

    print("\nTesting praesepe gyro model...")
    test_gyro_model_praesepe()

    print("\nTesting praesepe gyro model...")
    test_praesepe_gyro_model()

    print("\nTesting age model...")
    test_age_model()

    print("\nTesting sigma...")
    test_sigma()

    print("\nTesting B-V calculation...")
    test_calc_bv()

    print("\nTesting likelihood function on the Sun...")
    test_lnprob_higher_likelihood_sun()

    print("\nTesting likelihood function on data...")
    test_lnprob_higher_likelihood_real()

    print("\nTesting convective overturn timescale calculation...")
    test_convective_overturn_timescale()

    # print("\n Test for NaNs")
    # test_for_nans()
