"""
STARDATE
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
import pandas as pd
from isochrones.mist import MIST_Isochrone
mist = MIST_Isochrone()
from isochrones import StarModel
import emcee
import h5py


def gyro_model(log10_age, bv):
    """
    Given a B-V colour and an age, predict a rotation period.
    params:
    -------
    log10_age: (array)
        The log age of a star: log10(age) in years.
    bv: (array)
        The B-V colour of a star.
    returns:
    --------
    Rotation period in days.
    """
    age_myr = (10**log10_age)*1e-6

    a, b, c, n = [.4, .31, .45, .55]

    log_P = n*np.log10(age_myr) + np.log10(a) + b*np.log10(bv-c)
    return 10**log_P


def gyro_model_praesepe(log10_age, bprp):
    """
    Given a G_BP-G_RP colour and an age, predict a rotation period.
    parameters using a model fit to Praesepe:
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


def gyro_model_rossby(log10_age, bv, mass, Ro_cutoff=2.16, rossby=True):
    """
    Predict a rotation period from an age, color and mass using a combination
    of the Angus et al. (2015) gyrochronology model with the
    van Saders et al. (2016) weakened magnetic braking correction.
    params:
    -------
    log10_age: (float)
        The log10_age of a star in years.
    bv: (float)
        The B-V color of a star.
    mass: (float)
        The mass of a star in Solar masses.
    Ro_cutoff: (float, optional)
        The critical Rossby number after which stars retain their rotation
        period. This is 2.16 in van Saders et al. (2016) and 2.08 in van
        Saders et al. (2018). We adopt 2.16.
    rossby: (bool, optional)
        If True (default), the van Saders (2016) weakened magnetic braking law
        will be implemented. If false, the Angus et al. (2015) gyrochronology
        relation will be used unmodified.
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
    """
    Calculate B-V colour from stellar parameters using the MIST isochrones.
    params:
    ------
    mag_pars: (list)
        A list containing EEP, age, feh, distance, Av for a star.
    returns:
    -------
        B-V color. (float)
    """
    B = mist.mag["B"](*mag_pars)
    V = mist.mag["V"](*mag_pars)
    return B-V


def lnprior(params):
    """
    lnprior on all parameters.
    params need to be linear except age which is log10(age [yr]).
    params:
    -------
    params: (array)
        An array of EEP, age, feh, distance and extinction.
    returns:
    -------
    The prior probability for these parameters.
    """

    # finite_mask = np.isfinite(params)
    # if sum(finite_mask) < len(params):
    #     print(params, "non-finite parameter")

    # log Priors over age, metallicity and distance.
    # (The priors in priors.py are not in log)
    age_prior = np.log(priors.age_prior(params[1]))
    feh_prior = np.log(priors.feh_prior(params[2]))
    distance_prior = np.log(priors.distance_prior(params[3]))

    # Uniform prior on extinction.
    mAv = (0 <= params[4]) * (params[4] < 1)  # Prior on A_v
    mAv &= np.isfinite(params[4])
    mAv = mAv == 1

    # Uniform prior on EEP
    m = (190 < params[0]) * (params[0] < 500)  # Broad bounds on EEP.
    m &= np.isfinite(params[0])


    if mAv and m and np.isfinite(age_prior) and np.isfinite(distance_prior) \
            and np.isfinite(feh_prior):
        return age_prior + feh_prior + distance_prior

    else:
        return -np.inf


def lnprob(lnparams, *args):
    """
    The ln-probability function.
    params:
    ------
    lnparams: (array)
        The parameter array: [EEP, log10(age [yrs]), [Fe/H], ln(distance [kpc]), A_v]
    *args: (list)
        The arguments. A list containing
        [mod, period, period_err, bv, mass, iso_only, gyro_only].
        mod is the isochrones starmodel object which is set up in stardate.py.
        period, period_err, bv and mass are the rotation period and rotation
        period uncertainty (in days), B-V color and mass [M_sun].
        bv and mass should both be None unless gyrochronology only is being
        used.
    """

    # Transform mass and distance back to linear.
    params = lnparams*1
    params[3] = np.exp(lnparams[3])

    # Unpack the args.
    mod, period, period_err, bv, mass, iso_only, gyro_only = args

    # If the prior is -inf, don't even try to calculate the isochronal
    # likelihood.
    lnpr = mod.lnprior(params)
    if not np.isfinite(lnpr):
        return lnpr, lnpr

    # If isochrones only, just return the isochronal lhf.
    if iso_only:
        return mod.lnlike(params) + lnpr, lnpr

    # If a B-V is not provided, calculate it.
    if bv == None:
        assert gyro_only == False, "You must provide a B-V colour if you "\
            "want to calculate an age using gyrochronology only."
        bv = calc_bv(params)

    # If the B-V value calculated is nan, return the prior.
    if not np.isfinite(bv):
        return lnpr, lnpr

    # Check that the period is a positive, finite number. It doesn't matter
    # too much what the lhf is here, as long as it is constant.
    if not period or not np.isfinite(period) or period <= 0.:
        gyro_lnlike = -.5*((5/(20.))**2) - np.log(20.)

    # If FGK and MS:
    elif bv > .45 and params[0] < 454:

        if not mass:  # If a mass is not provided, calculate it.
            mass = mist.mass(params[0], params[1], params[2])

        # Calculate a period using the gyrochronology model
        period_model = gyro_model_rossby(params[1], bv, mass)

        # Calculate the gyrochronology likelihood.
        gyro_lnlike = -.5*((period - period_model) / (period_err))**2 \
            - np.log(period_err)

    # If hot, use a broad log-gaussian model with a mean of .5 for rotation.
    elif bv < .45:
        if not mass:
            mass = mist.mass(params[0], params[1], params[2])
        period_model = .5  # 1
        # gyro_lnlike = -.5*((period - period_model)
        #                     / (period_err+10))**2 - np.log(period_err+10)
        gyro_lnlike = -.5*((np.log10(period) - period_model)
                            / (.55))**2 - np.log(.55)

    # If evolved, use a gyrochronology relation with inflated uncertainties.
    elif bv > .45 and params[0] >= 454:
        if not mass:
            mass = mist.mass(params[0], params[1], params[2])
        period_model = gyro_model_rossby(params[1], bv, mass)
        gyro_lnlike = -.5*((period - period_model) / (period_err+10))**2 \
            - np.log(period_err+10)

    if gyro_only:
        return gyro_lnlike + lnpr, lnpr

    return mod.lnlike(params) + gyro_lnlike + lnpr, lnpr


def nll(lnparams, *args):
    """
    The negative log likelihood.
    lnparams are [eep, log10(age [yrs]), [Fe/H], ln(distance [kpc]), A_v]
    """

    # Transform mass and distance back to linear.
    params = lnparams*1
    params[3] = np.exp(lnparams[3])

    mod, period, period_err, iso_only, rossby = args
    bv = calc_bv(params)

    # If isochrones only, just return the isochronal lhf.
    if iso_only:
        return - mod.lnlike(params)

    mass = mist.mass(params[0], params[1], params[2])
    gyro_lnlike = -.5*((period - gyro_model_rossby(params[1], bv, mass,
                                                   rossby)) / period_err)**2 \
        - np.log(period_err)

    return - mod.lnlike(params) - gyro_lnlike


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
