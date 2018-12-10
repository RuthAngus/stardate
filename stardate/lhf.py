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
from . import priors


def gyro_model(log10_age, bv):
    """
    Given a B-V colour and an age, predict a rotation period.
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


def gyro_model_praesepe(log10_age, bprp):
    """
    Given a G_BP-G_RP colour and an age, predict a rotation period.
    parameters:
    ----------
    log10_age: (array)
        The log age of a star: log10(age) in years.
    bprp: (array)
        The Bp-rp colour of a star.
    """
    age_gyr = (10**log10_age)*1e-9
    log_age_gyr = np.log10(age_gyr)
    log_c = np.log10(bprp)
    params = [1.10469903, 0.6183025, -4.452133, 31.02877576, -47.76497323,
              0.63604919]
    log_P = params[0] + params[1]*log_c + params[2]*log_c**2 \
        + params[3]*log_c**3 + params[4]*log_c**4 + params[5]*log_age_gyr
    return 10**log_P


def gyro_model_rossby(log10_age, bv, mass, rossby=True, Ro_cutoff=2.16):
    """
    Predict a rotation period from an age and color (and mass if the rossby
    cutoff model is used).
    params:
    -------
    args: (list)
        Either containing [log10_age, bv] in which case the standard gyro
        model will be used.
        Or [mass, log10_age, bv] in which case the Rossby number cutoff model
        will be used.
    Ro_cutoff: (float, optional)
        The critical Rossby number after which stars retain their rotation
        period.
        This is 2.16 in van Saders et al. (2016) and 2.08 in van Saders et al.
        (2018).
    """
    # Angus et al. (2015) parameters.
    a, b, c, n = [.4, .31, .45, .55]

    age_myr = (10**log10_age)*1e-6

    if not rossby:  # If Rossby model is switched off
        # Standard gyro model
        log_P = n*np.log10(age_myr) + np.log10(a) + b*np.log10(bv-c)
        return 10**log_P

    # Otherwise the Rossby model is switched on.
    # Calculate the maximum theoretical rotation period for this mass.
    pmax = Ro_cutoff * convective_overturn_time(mass)

    # Calculate the age this star reaches pmax, based on its B-V color.
    age_thresh_myr = (pmax/(a*(bv-c)**b))**(1./n)
    log10_age_thresh = np.log10(age_thresh_myr*1e6)

    # If star younger than critical age, predict rotation from age and color.
    if log10_age < log10_age_thresh:
        log_P = n*np.log10(age_myr) + np.log10(a) + b*np.log10(bv-c)

    # If star older than this age, return maximum possible rotation period.
    else:
        log_P = np.log10(pmax)
    return 10**log_P


def calc_bv(mag_pars):
    # Calculate B-V
    B = mist.mag["B"](*mag_pars)
    V = mist.mag["V"](*mag_pars)
    return B-V


def lnprior(params):
    """
    lnprior on all parameters.
    params need to be linear except age which is log10(age [yr]).
    """

    # log Priors over age, metallicity and distance.
    # (The priors in priors.py are not in log)
    age_prior = np.log(priors.age_prior(params[1]))
    feh_prior = np.log(priors.feh_prior(params[2]))
    distance_prior = np.log(priors.distance_prior(params[3]))

    # Uniform prior on extinction.
    mAv = (0 <= params[4]) * (params[4] < 1)  # Prior on A_v
    mAv = mAv == 1

    # Uniform prior on EEP
    m = (0 < params[0]) * (params[0]) < 10000  # Broad bounds on mass.

    if mAv and m and np.isfinite(age_prior) and np.isfinite(distance_prior):
        return age_prior + feh_prior + distance_prior

    else:
        return -np.inf


def lnprob(lnparams, *args):
    """
    The ln-probability function.
    lnparams are [eep, log10(age [yrs]), [Fe/H], ln(distance [kpc]), A_v]
    If EEP is greater than 425, the star has started evolving up the
    subgiant branch, so it should have a precise isochronal age and an
    unreliable gyro age -- shut gyrochronology off!
    If the Rossby number is greater than 2.16, shut gyrochronology off.
    """

    # Transform mass and distance back to linear.
    params = lnparams*1
    params[3] = np.exp(lnparams[3])

    mod, period, period_err, iso_only, rossby = args
    bv = calc_bv(params)

    # If the prior is -inf, don't even try to calculate the isochronal
    # likelihood.
    # lnpr = mod.lnprior(params)
    lnpr = lnprior(params)
    if not np.isfinite(lnpr):
        return lnpr, lnpr

    # If isochrones only, just return the isochronal lhf.
    if iso_only:
        return mod.lnlike(params) + lnpr, lnpr

    # Check that the period is a positive, finite number. It doesn't matter
    # too much what the lhf is here, as long as it is constant.
    if not period or not np.isfinite(period) or period <= 0.:
        gyro_lnlike = -.5*((5/(20.))**2) - np.log(20.)

    # If cool and MS:
    elif bv > .45 and params[0] < 454:
        mass = mist.mass(params[0], params[1], params[2])
        gyro_lnlike = -.5*((period
                            - gyro_model_rossby(params[1], bv, mass, rossby))
                            / period_err)**2 - np.log(period_err)

    # If evolved or hot, use a broad gaussian model for rotation.
    else:
        # gyro_lnlike = -.5*(((np.log10(period) - .5)/.55)**2) \
        #     - np.log(.55)
        gyro_lnlike = -.5*(((np.log10(period) - .5)/.55)**2) \
            - np.log(.55)
        # gyro_lnlike = -.5*((period - 5)/(period_err*20.))**2 \
        #     - np.log(20.*period_err)
#         gyro_lnlike = -.5*((period - .5)/(period_err*100))**2 \
#            - np.log(100*period_err)

    return lnpr, lnpr
    # return mod.lnlike(params) + gyro_lnlike + lnpr, lnpr


def convective_overturn_time(*args):
    """
    Estimate the convective overturn time using equation 11 in Wright et al.
    (2011): https://arxiv.org/abs/1109.4634
    log tau = 1.16 - 1.49log(M/M⊙) - 0.54log^2(M/M⊙)
    (I assume log is log10)
    params:
    ------
    EITHER:
    mass: (float)
        Mass in Solar units
    OR
    eep: (float)
        The Equivalent evolutionary point of a star. 355 for the Sun.
    age: (float)
        The age of a star in log_10(years).
    feh: (float)
        The metallicity of a star.
    """

    if len(args) > 1:
        # Convert eep, age and feh to mass (mass will be in Solar mass units)
        eep, age, feh = args
        M = mist.mass(eep, age, feh)
    else:
        M = args[0]

    log_tau = 1.16 - 1.49*np.log10(M) - .54*(np.log10(M))**2
    return 10**log_tau


def run_mcmc(obs, args, p_init, backend, ndim=5, nwalkers=24, thin_by=100,
             max_n=100000):
    max_n = max_n//thin_by

    # p0 = [p_init + np.random.randn(ndim)*1e-4 for k in range(nwalkers)]

    # Broader gaussian for EEP initialization
    p0 = np.empty((nwalkers, ndim))
    p0[:, 0] = np.random.randn(nwalkers)*10 + p_init[0]
    p0[:, 1] = np.random.randn(nwalkers)*1e-4 + p_init[1]
    p0[:, 2] = np.random.randn(nwalkers)*1e-4 + p_init[2]
    p0[:, 3] = np.random.randn(nwalkers)*1e-4 + p_init[3]
    p0[:, 4] = np.random.randn(nwalkers)*1e-4 + p_init[4]
    p0 = list(p0)

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
    for sample in sampler.sample(p0, iterations=max_n, thin_by=thin_by,
                                 store=True, progress=True):
        # Only check convergence every 100 steps
        # if sampler.iteration % 100:
        #     continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0) * thin_by
        autocorr[index] = np.mean(tau)
        index += 1

        # # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
    # ======================================================================

    return sampler


# def make_plots(sampler, i, truths, savedir, burnin=10000):

#     nwalkers, nsteps, ndim = np.shape(sampler.chain)
#     assert burnin < nsteps, "The number of burn in samples to throw away" \
#         "can't exceed the total number of samples."

#     samples = sampler.flatchain

#     print("Plotting age posterior")
#     age_gyr = (10**samples[burnin:, 1])*1e-9
#     plt.hist(age_gyr)
#     plt.xlabel("Age [Gyr]")
#     med, std = np.median(age_gyr), np.std(age_gyr)
#     if truths[1]:
#         plt.axvline(10**(truths[1])*1e-9, color="tab:orange",
#                     label="$\mathrm{True~age~[Gyr]}$")
#     plt.axvline(med, color="k", label="$\mathrm{Median~age~[Gyr]}$")
#     plt.axvline(med - std, color="k", linestyle="--")
#     plt.axvline(med + std, color="k", linestyle="--")
#     plt.savefig("{0}/{1}_marginal_age".format(savedir, str(i).zfill(4)))
#     plt.close()

#     print("Plotting production chains...")
#     plt.figure(figsize=(16, 9))
#     for j in range(ndim):
#         plt.subplot(ndim, 1, j+1)
#         plt.plot(sampler.chain[:, burnin:, j].T, "k", alpha=.1)
#     plt.savefig("{0}/{1}_chains".format(savedir, str(i).zfill(4)))
#     plt.close()

#     print("Making corner plot...")
#     labels = ["$\mathrm{EEP}$",
#               "$\log_{10}(\mathrm{Age~[yr]})$",
#               "$\mathrm{[Fe/H]}$",
#               "$\ln(\mathrm{Distance~[Kpc])}$",
#               "$A_v$"]
#     corner.corner(samples[burnin:, :], labels=labels, truths=truths);
#     plt.savefig("{0}/{1}_corner".format(savedir, str(i).zfill(4)))
#     plt.close()

#     # Make mass histogram
#     samples = sampler.flatchain
#     mass_samps = mist.mass(samples[:, 0], samples[:, 1], samples[:, 2])
#     plt.hist(mass_samps, 50);
#     if truths[0]:
#             plt.axvline(truths[0], color="tab:orange",
#                         label="$\mathrm{True~mass~}[M_\odot]$")
#     plt.savefig("{0}/{1}_marginal_mass".format(savedir, str(i).zfill(4)))
#     plt.close()
