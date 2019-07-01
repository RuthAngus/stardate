"""
Make sure the isochronal likelihood function looks the same as stardate with
isochrones only
"""

import emcee
import numpy as np
from stardate.lhf import lnprob, lnlike
import stardate as sd
import time
from isochrones import SingleStarModel, get_ichrone
bands = ["B", "V", "J", "H", "K", "BP", "RP"]
mist = get_ichrone("mist", bands=bands)


def iso_lnprob(params, *args):
    linparams = params*1
    linparams[3] = np.exp(linparams[3])
    mod, iso_params = args
    like, prior = mod.lnlike(linparams), mod.lnprior(linparams)
    prob = like + prior
    if not np.isfinite(prob):
        prob = -np.inf
    return prob


def test_iso_lnlike():

    iso_params = {"teff": (5770, 10),
                "feh": (0, .01),
                "logg": (4.44, .1),
                "parallax": (1, 1)}

    mod = SingleStarModel(mist, **iso_params)  # StarModel isochrones obj
    params = [354, np.log10(4.56*1e9), 0., 1000, 0.]
    lnparams = [354, np.log10(4.56*1e9), 0., np.log(1000), 0.]
    lnpr = mod.lnprior(params)
    # lnparams = [350, 9, 0., 6, 0.]
    args1 = [mod, iso_params]

    # Calculate the lnprob above
    iso_lp = iso_lnprob(lnparams, *args1)
    start = time.time()
    for i in range(100):
        iso_lp = iso_lnprob(lnparams, *args1)
    end = time.time()

    # Calculate the stardate lnprob
    args2 = [mod, None, None, True, False, True, "praesepe"]

    start = time.time()
    for i in range(100):
        lp = lnprob(lnparams, *args2)[0]
    end = time.time()
    # print("time = ", end - start)

    ll = lnlike(lnparams, *args2)
    lnprior = lp - ll

    assert iso_lp == lp
    assert np.isclose(iso_lp - lnpr, ll)
    assert lnpr == lnprior

    # THIS TEST
    np.random.seed(42)
    nwalkers, ndim, nsteps = 50, 5, 100
    p0 = [np.random.randn(ndim)*1e-4 + lnparams for j in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, iso_lnprob, args=args1);

    start = time.time()
    sampler.run_mcmc(p0, nsteps);
    end = time.time()

    samples = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))
    test_median = np.median(samples[:, 1])

    # STARDATE
    star = sd.Star(iso_params, filename="sun_test")

    start = time.time()
    star.fit(max_n=100, inits=params, thin_by=1, seed=42, save_samples=False)
    end = time.time()

    iso_median = np.median(star.samples[:, 1])
    assert iso_median == test_median


if __name__ == "__main__":
    test_iso_lnlike()
