import numpy as np
import pandas as pd
from lhf import lnprob, calc_bv, gyro_model
from lhf import convective_overturn_time, gyro_model_praesepe
from lhf import gyro_model_rossby
# import stardate
from isochrones.mist import MIST_Isochrone
from isochrones import StarModel
mist = MIST_Isochrone()


def good_vs_bad(good_lnprob, good_lnparams, args, nsamps):
    """
    Compare the likelihood of nsamps random parameter values against the
    likelihood of the true values.
    The true values should be higher.
    """
    for i in range(nsamps):
        bad_lnparams = good_lnparams*1
        bad_lnparams[0] = np.random.randn(1) * 10 + good_lnparams[0]  # eep
        bad_lnparams[1] = np.random.randn(1) * .5 + good_lnparams[1]  # age
        bad_lnparams[2] = np.random.randn(1) * .05 + good_lnparams[2]  # feh
        bad_lnparams[3] = np.random.randn(1) * .1 + good_lnparams[3]  # dist
        bad_lnprob = lnprob(bad_lnparams, *args)
        assert bad_lnprob[0] < good_lnprob[0], \
            "True parameters values must give a higher likelihood than" \
            " wrong values"


def test_lnprob_higher_likelihood_sun():
    """
    Make sure the likelihood goes down when the parameters are a worse fit.
    Test on Solar values.
    """
    iso_params = pd.DataFrame(dict({"teff": (5777, 10),
                                "logg": (4.44, .05),
                                "feh": (0., .001),
                                "parallax": (1., .01)}))  # mas

    # Set up the StarModel isochrones object.
    mod = StarModel(mist, **iso_params)
    args = [mod, 26., 1., False, True]  # the lnprob arguments]

    good_lnparams = [346, np.log10(4.56*1e9), 0., np.log(1000), 0.]
    good_lnprob = lnprob(good_lnparams, *args)
    good_vs_bad(good_lnprob, good_lnparams, args, 10)


def test_lnprob_higher_likelihood_real():
    """
    The same test as above but for simulated data.
    """
    df = pd.read_csv("../paper/code/data/simulated_data.csv")
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
    args = [mod, df.prot[i], 1, False, True]  # the lnprob arguments]
    good_lnparams = [df.eep.values[i], df.age.values[i], df.feh.values[i],
                     np.log(df.d_kpc.values[i]*1e3), df.Av.values[i]]
    good_lnprob = lnprob(good_lnparams, *args)
    good_vs_bad(good_lnprob, good_lnparams, args, 10)


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

    args = [mod, None, None, False, True]  # the lnprob arguments]
    none_lnprob = lnprob(lnparams, *args)

    args = [mod, np.nan, np.nan, False, True]  # the lnprob arguments]
    nan_lnprob = lnprob(lnparams, *args)

    args = [mod, 0., 0., False, True]  # the lnprob arguments]
    zero_lnprob = lnprob(lnparams, *args)

    args = [mod, 26., 1., True, True]  # the lnprob arguments]
    iso_lnprob = lnprob(lnparams, *args)

    args = [mod, 26., 1., False, True]  # the lnprob arguments]
    gyro_lnprob = lnprob(lnparams, *args)

    # check that gyro is switched off for all of these.
    none_lnprob == nan_lnprob
    nan_lnprob == zero_lnprob

    # check that gyro on gives different lnprob
    assert gyro_lnprob != iso_lnprob

    # Likelihood should be greater for dwarfs because gyro lnlike is a broad
    # Gaussian for giants.
    giant_params = [455, np.log10(4.56*1e9), 0., np.log(1000), 0.]
    dwarf_params = [453, np.log10(4.56*1e9), 0., np.log(1000), 0.]
    args = [mod, 26., 1., False, True]
    giant_lnprob = lnprob(giant_params, *args)
    dwarf_lnprob = lnprob(dwarf_params, *args)
    assert giant_lnprob < dwarf_lnprob

    # Likelihood should be greater for cool stars because gyro lnlike is a
    # broad Gaussian for giants.
    heep, hage, hfeh = 405, np.log10(2.295*1e9), 0.
    ceep, cage, cfeh = 355, np.log10(4.56*1e9), 0.
    hot_params = [heep, hage, hfeh, np.log(1000), 0.]
    cool_params = [ceep, cage, cfeh, np.log(1000), 0.]
    cool_prot = gyro_model_rossby(cage, calc_bv(cool_params),
                                  mist.mass(ceep, cage, cfeh))
    hot_prot = gyro_model_rossby(hage, calc_bv(hot_params),
                                  mist.mass(heep, hage, hfeh))
    cool_args = [mod, cool_prot, 1., False, True]
    hot_args = [mod, hot_prot, 1., False, True]

    hot_lnprob = lnprob(hot_params, *args)
    cool_lnprob = lnprob(cool_params, *args)
    assert hot_lnprob[0] < cool_lnprob[0], "cool star likelihood should be" \
        " higher than hot star likelihood"


def test_calc_bv():
    sun = [355, np.log10(4.56*1e9), 0., np.log(1000), 0.]
    bv_sun = calc_bv(sun)
    assert .6 < bv_sun
    assert bv_sun < .7


def test_convective_overturn_timescale():
    tau = convective_overturn_time(355, np.log10(4.56*1e9), 0.)
    tau = convective_overturn_time(1)
    solarRo = 2.16  # van Saders 2016 (2.08 in van Saders 2018)
    solarProt = 26
    solartau = solarProt / solarRo
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


def test_praesepe_gyro_model():
    prot = gyro_model_praesepe(np.log10(4.56*1e9), .82)
    assert 24.5 < prot < 27.5


def test_gyro_model_rossby():
    """
    Make sure that the rossby model gives Solar values for the Sun.
    Make sure it gives a maximum rotation period of pmax.
    """
    age = np.log10(4.56*1e9)
    sun = [355, age, 0., np.log(1000), 0.]
    prot_sun = gyro_model_rossby(age, .65, 1.)
    assert 24 < prot_sun
    assert prot_sun < 27

    prot_8_sun = gyro_model_rossby(np.log(8*1e9), .65, 1.)
    assert 26 < prot_8_sun
    assert prot_8_sun < 32

    prot_10_sun = gyro_model_rossby(np.log(10*1e9), .65, 1.)
    assert prot_10_sun == prot_8_sun


if __name__ == "__main__":
    print("Testing gyro model with Rossby switch...")
    test_gyro_model_rossby()

    # print("\nTesting likelihood function behaviour...")
    # test_likelihood_rotation_giant()

    print("\nTesting B-V calculation...")
    test_calc_bv()

    print("\nTesting the Praesepe gyro model...")
    test_praesepe_gyro_model()

    print("\nTesting likelihood function on the Sun...")
    test_lnprob_higher_likelihood_sun()

    print("\nTesting likelihood function on data...")
    test_lnprob_higher_likelihood_real()

    print("\nTesting convective overturn timescale calculation...")
    test_convective_overturn_timescale()

    print("\nTesting original gyro model...")
    test_gyro_model()
