import numpy as np
import pandas as pd
from stardate.lhf import lnprob
import stardate
from isochrones.mist import MIST_Isochrone
from isochrones import StarModel
mist = MIST_Isochrone()


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
    args = [mod, 26., 1., False]  # the lnprob arguments]

    bad_lnparams = [355, np.log10(4.56*1e9), 0.2, np.log(1000), 0.]
    good_lnparams = [355, np.log10(4.56*1e9), 0., np.log(1000), 0.]

    bad_lnprob = lnprob(bad_lnparams, *args)
    good_lnprob = lnprob(good_lnparams, *args)

    print(bad_lnprob, good_lnprob)

    assert bad_lnprob[0] < good_lnprob[0], \
        "True parameters values must give a higher likelihood than wrong"\
        " values"

def test_lnprob_higher_likelihood_real():
    # Now test on real data
    df = pd.read_csv("simulated_data.csv")
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
    iso_params = pd.DataFrame(dict({"teff": (df.teff[i], teff_err),
                                    "logg": (df.logg[i], logg_err),
                                    "feh": (df.feh[i], feh_err),
                                    "jmag": (df.jmag[i], jmag_err),
                                    "hmag": (df.hmag[i], hmag_err),
                                    "kmag": (df.kmag[i], kmag_err),
                                    "parallax": (df.parallax[i],
                                                 parallax_err)}))

    # Set up the StarModel isochrones object.
    mod = StarModel(mist, **iso_params)
    args = [mod, df.prot[i], 1, False]  # the lnprob arguments]

    lnparams = [df.eep.values[i], df.age.values[i], df.feh.values[i],
                np.log(df.d_kpc.values[i]*1e3) + .2, df.Av.values[i]]
    bad_lnprob = lnprob(lnparams, *args)
    lnparams = [df.eep.values[i], df.age.values[i], df.feh.values[i],
                np.log(df.d_kpc.values[i]*1e3), df.Av.values[i]]
    good_lnprob = lnprob(lnparams, *args)
    assert bad_lnprob[0] < good_lnprob[0], \
        "True parameters values must give a higher likelihood than wrong"\
        " values"


def test_likelihood_rotation():
    """
    Make sure that the lhf can cope with zeros, NaNs and None values for the
    rotation period.
    """
    iso_params = pd.DataFrame(dict({"teff": (5777, 10),
                                "logg": (4.44, .05),
                                "feh": (0., .001),
                                "parallax": (1., .01)}))  # mas
    # Set up the StarModel isochrones object.
    mod = StarModel(mist, **iso_params)
    lnparams = [355, np.log10(4.56*1e9), 0., np.log(1000), 0.]

    args = [mod, None, None, False]  # the lnprob arguments]
    none_lnprob = lnprob(lnparams, *args)

    args = [mod, np.nan, np.nan, False]  # the lnprob arguments]
    nan_lnprob = lnprob(lnparams, *args)

    args = [mod, 0., 0., False]  # the lnprob arguments]
    zero_lnprob = lnprob(lnparams, *args)

    args = [mod, 26., 1., True]  # the lnprob arguments]
    iso_lnprob = lnprob(lnparams, *args)

    args = [mod, 26., 1., False]  # the lnprob arguments]
    gyro_lnprob = lnprob(lnparams, *args)

    # check that gyro is switched off for all of these.
    assert iso_lnprob == none_lnprob
    assert iso_lnprob == nan_lnprob
    assert iso_lnprob == zero_lnprob

    # check that gyro on gives different lnprob
    assert gyro_lnprob != iso_lnprob

    giant_params = [450, np.log10(4.56*1e9), 0., np.log(1000), 0.]
    args = [mod, 26., 1., True]
    iso_giant_lnprob = lnprob(giant_params, *args)
    args = [mod, 26., 1., False]
    gyro_giant_lnprob = lnprob(giant_params, *args)

    # These should be the same because gyro is switched off for giants.
    assert iso_giant_lnprob == gyro_giant_lnprob

if __name__ == "__main__":
    test_lnprob_higher_likelihood_sun()
    test_lnprob_higher_likelihood_real()
    test_likelihood_rotation()
