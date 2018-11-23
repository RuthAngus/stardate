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
import corner
import h5py


def setup(obs, gyro_only=False, iso_only=False):
    """
    DEPRECATED!!!!!
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


def lnprob(lnparams, *args):
    """
    lnparams are [eep, log10(age [yrs]), [Fe/H], ln(distance [kpc]), A_v]
    """

    # Transform mass and distance back to linear.
    params = lnparams*1
    # params[0] = np.exp(lnparams[0])
    params[3] = np.exp(lnparams[3])

    mod, period, period_err, iso_only = args

    # mag_pars = (params[0], params[1], params[2], params[3]*1e3, params[4])
    mag_pars = (params[0], params[1], params[2], params[3], params[4])
    B = mist.mag["B"](*mag_pars)
    V = mist.mag["V"](*mag_pars)
    bv = B-V

    # If the prior is -inf, don't even try to calculate the isochronal
    # likelihood.
    lnpr = mod.lnprior(params)
    if not np.isfinite(lnpr):
        return lnpr, lnpr

    if iso_only:
        return mod.lnlike(params) + lnpr, lnpr

    if bv > .45:
        gyro_lnlike = -.5*((period - gyro_model(params[1], bv))
                            /period_err)**2
    else:
        gyro_lnlike = 0

    # If EEP is greater than 425, the star has started evolving up the
    # subgiant branch, so it should have a precise isochronal age and an
    # unreliable gyro age -- shut gyrochronology off!
    # if params[0] > 425:
    #     gyro_lnlike = 0

    return mod.lnlike(params) + gyro_lnlike + lnpr, lnpr


def run_mcmc(obs, args, p_init, backend, ndim=5, nwalkers=24, max_n=100000):

    p0 = [p_init + np.random.randn(ndim)*1e-4 for k in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args,
                                    backend=backend)

    # Copied from https://emcee.readthedocs.io/en/latest/tutorials/monitor/
    # ======================================================================

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(p0, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # print("autocorrelation time = ", tau, "steps = ", sampler.iteration)
        # # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
    # ======================================================================

    return sampler


def make_plots(sampler, i, truths, savedir, burnin=10000):

    nwalkers, nsteps, ndim = np.shape(sampler.chain)
    assert burnin < nsteps, "The number of burn in samples to throw away" \
        "can't exceed the total number of samples."

    samples = sampler.flatchain

    print("Plotting age posterior")
    age_gyr = (10**samples[burnin:, 1])*1e-9
    plt.hist(age_gyr)
    plt.xlabel("Age [Gyr]")
    med, std = np.median(age_gyr), np.std(age_gyr)
    if truths[1]:
        plt.axvline(10**(truths[1])*1e-9, color="tab:orange",
                    label="$\mathrm{True~age~[Gyr]}$")
    plt.axvline(med, color="k", label="$\mathrm{Median~age~[Gyr]}$")
    plt.axvline(med - std, color="k", linestyle="--")
    plt.axvline(med + std, color="k", linestyle="--")
    plt.savefig("{0}/{1}_marginal_age".format(savedir, str(i).zfill(4)))
    plt.close()

    print("Plotting production chains...")
    plt.figure(figsize=(16, 9))
    for j in range(ndim):
        plt.subplot(ndim, 1, j+1)
        plt.plot(sampler.chain[:, burnin:, j].T, "k", alpha=.1)
    plt.savefig("{0}/{1}_chains".format(savedir, str(i).zfill(4)))
    plt.close()

    print("Making corner plot...")
    labels = ["$\mathrm{EEP}$",
              "$\log_{10}(\mathrm{Age~[yr]})$",
              "$\mathrm{[Fe/H]}$",
              "$\ln(\mathrm{Distance~[Kpc])}$",
              "$A_v$"]
    corner.corner(samples[burnin:, :], labels=labels, truths=truths);
    plt.savefig("{0}/{1}_corner".format(savedir, str(i).zfill(4)))
    plt.close()

    # Make mass histogram
    samples = sampler.flatchain
    mass_samps = mist.mass(samples[:, 0], samples[:, 1], samples[:, 2])
    plt.hist(mass_samps, 50);
    if truths[0]:
            plt.axvline(truths[0], color="tab:orange",
                        label="$\mathrm{True~mass~}[M_\odot]$")
    plt.savefig("{0}/{1}_marginal_mass".format(savedir, str(i).zfill(4)))
    plt.close()
