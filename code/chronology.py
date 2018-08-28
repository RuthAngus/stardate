"""
CHRONOLOGY
=====================
This package produces posterior PDFs over age, mass, bulk metalicity,
distance and V-band extinction for stars from their spectroscopic parameters
(T_eff, log g and observed bulk metallicity), their apparent magnitudes in
a range of bandpasses, their parallaxes and rotation periods, if available.
The minimum requirements for producing age estimates are photometric colors,
however the addition of extra information improves the constraint on the
stellar properties. In particular, this method is designed to incorporate a
gyrochronology model into standard isochrone fitting. If you do not have
rotation periods for your stars, the posteriors over parameters produced by
this code will be very similar to those produce using the isochrones.py
package on its own.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from isochrones.mist import MIST_Isochrone
mist = MIST_Isochrone()
from isochrones import StarModel
import emcee
import priors
import corner
import h5py


def setup(obs, gyro_only=False, iso_only=False):
    """
    setup: (function)
        Generates the StarModel object with observational inputs and the
        arguments to pass to the log-probability function.
    params:
    -------
    obs: (dataframe)
        A single row of a pandas dataframe containing J-band, H-band and
        K-band magnitudes, T_eff, logg, feh, parallax, rotation period and
        precomputed B-V colors. It also contains observational uncertainties
        for all these observables.
        If T_eff, logg, feh or parallax are unavailable, obs may contain
        "None" in their places.
    gyro_only: (boolean)
        True if only the gyrochronal likelihood is to be used.
        Default = False.
    iso_only: (boolean)
        True if only the isochronal likelihood is to be used.
        Default = False.
    returns:
    --------
    mod: (starmodel object)
        The starmodel object.
    param_dict: (dictionary)
        A dictionary of the observed parameters which must be passed to the
        likelihood function.
    args: (list)
        A list of arguments that must be passed to the log-probability
        function.
    """

    # Set up the StarModel object needed to calculate the likelihood.
    param_dict = {"J": (obs.jmag, obs.jmag_err),
                  "H": (obs.hmag, obs.hmag_err),
                  "K": (obs.kmag, obs.kmag_err),
                  "teff": (obs.teff, obs.teff_err),
                  "logg": (obs.logg, obs.logg_err),
                  "feh": (obs.feh, obs.feh_err),
                  "parallax": (obs.parallax, obs.parallax_err)  # Isochrones.py takes milliarcseconds
                  }

    mod = StarModel(mist, **param_dict)  # Set up the StarModel isochrones object.
    args = [mod, obs.prot, obs.prot_err, obs.BV, gyro_only, iso_only]  # the lnprob arguments
    return mod, param_dict, args


def gyro_model(log10_age, bv):
    """
    Given a B-V colour and an age, predict a rotation period.
    Returns log(age) in Myr.
    parameters:
    ----------
    log10_age: (array)
        The log age of a star: log10(age) in years.
    bv: (array)
        The B-V colour of a star.
    """
    age_myr = (10**log10_age)*1e-6

    a, b, c, n = [.4, .31, .45, .55]

    log_P = n*np.log10(age_myr) + np.log10(a) + b*np.log10(bv-c)
    return 10**log_P


def lnprior(params):
    """
    lnprior on all parameters.
    params need to be linear except age which is log10(age [yr]).
    """

    # log Priors over age, metallicity and distance.
    # (The priors in priors.py are not in log)
    age_prior = np.log(priors.age_prior(params[1]))
    feh_prior = np.log(priors.feh_prior(params[2]))
    distance_prior = np.log(priors.distance_prior(np.exp(params[3])))

    # Uniform prior on extinction.
    mAv = (0 <= params[4]) * (params[4] < 1)  # Prior on A_v
    mAv = mAv == 1

    # Uniform prior on mass
    m = (-20 < params[0]) * (params[0]) < 20  # Broad bounds on mass.

    if mAv and m and np.isfinite(age_prior) and np.isfinite(distance_prior):
        return age_prior + feh_prior + distance_prior

    else:
        return -np.inf


def lnprob(lnparams, *args):
    """
    lnparams are [ln(mass), log10(age [yrs]), [Fe/H], ln(distance [kpc]), A_v]
    """

    # Transform mass and distance back to linear.
    params = lnparams*1
    params[0] = np.exp(lnparams[0])
    params[3] = np.exp(lnparams[3])

    mod, period, period_err, bv_est, gyro_only, iso_only = args

    mag_pars = (params[0], params[1], params[2], params[3]*1e3, params[4])
    B = mist.mag["B"](*mag_pars)
    V = mist.mag["V"](*mag_pars)
    bv = B-V

    # If the prior is -inf, don't even try to calculate the isochronal
    # likelihood.
    lnpr = lnprior(params)
    if lnpr == -np.inf:
        return lnpr

    else:

        if iso_only:
            return mod.lnlike(params) + lnpr

        else:
            if bv > .45:
                gyro_lnlike = -.5*((period - gyro_model(params[1], bv))
                                   /period_err)**2
            else:
                gyro_lnlike = 0

        # B-V is estimated from mass, etc, so you need to use a different B-V
        # estimate if gyro_only.
        if gyro_only:
            return -.5*((period - gyro_model(params[1], bv_est))
                        /period_err)**2 + lnpr

        else:
            return mod.lnlike(params) + gyro_lnlike + lnpr


def run_mcmc(obs, args, p_init, burnin=5000, production=10000, ndim=5,
             nwalkers=24):

    p0 = [p_init + np.random.randn(ndim)*1e-4 for k in range(nwalkers)]

    print("Burning in...")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)
    p0, lnp, state = sampler.run_mcmc(p0, burnin)

    print("Production run...")
    sampler.reset()
    p0, lnp, state = sampler.run_mcmc(p0, production)

    return sampler


def make_plots(sampler, i, truths, savedir):
    ndim = 5

    samples = sampler.flatchain

    print("Plotting age posterior")
    age_gyr = (10**samples[:, 1])*1e-9
    plt.hist(age_gyr)
    plt.xlabel("Age [Gyr]")
    med, std = np.median(age_gyr), np.std(age_gyr)
    plt.axvline(10**(truths[1])*1e-9, color="tab:orange",
                label="$\mathrm{True~age~[Gyr]}$")
    plt.axvline(med, color="k", label="$\mathrm{Median~age~[Gyr]}$")
    plt.axvline(med - std, color="k", linestyle="--")
    plt.axvline(med + std, color="k", linestyle="--")
    plt.savefig("{0}/{1}_marginal_age".format(savedir, i))
    plt.close()

    print("Plotting production chains...")
    plt.figure(figsize=(16, 9))
    for j in range(ndim):
        plt.subplot(ndim, 1, j+1)
        plt.plot(sampler.chain[:, :, j].T, "k", alpha=.1)
    plt.savefig("{0}/{1}_chains".format(savedir, i))
    plt.close()

    print("Making corner plot...")
    labels = ["ln(mass [M_sun])", "log10(age [yr])", "[Fe/H]",
              "ln(distance [Kpc])", "Av"]
    corner.corner(samples, labels=labels);
    plt.savefig("{0}/{1}_corner".format(savedir, i))
    plt.close()

    print("Making linear corner plot...")
    slin = samples*1
    slin[:, 0] = np.exp(samples[:, 0])
    slin[:, 3] = np.exp(samples[:, 3])
    slin[:, 1] = (10**samples[:, 1])*1e-9
    labels = ["mass [M_sun]", "age [Gyr]", "[Fe/H]", "distance [Kpc]", "Av"]
    corner.corner(slin, labels=labels);
    plt.savefig("{0}/{1}_corner_linear".format(savedir, i))
    plt.close()
