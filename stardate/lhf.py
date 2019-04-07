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
from isochrones import StarModel, get_ichrone
# mist = MIST_Isochrone(bands)
bands = ["B", "V", "J", "H", "K", "BP", "RP", "G"]
mist = get_ichrone("mist", bands=bands)
import emcee
import h5py


def gyro_model(log10_age, bv):
    """Predict a rotation period from an age and B-V colour.

    Given a B-V colour and an age, predict a rotation period using the Angus
    et al. (2015) gyrochronology model.

    Args:
        log10_age (float or array): The logarithmic age of a star or stars,
            log10(age), in years.
        bv (float or array): The B-V colour of a star or stars.

    Returns:
        Rotation period in days.

    """
    age_myr = (10**log10_age)*1e-6
    a, b, c, n = [.4, .31, .45, .55]
    hot = bv < .45
    if hot:
        return 0
    else:
        return (n*np.log10(age_myr) + np.log10(a) + b*np.log10(bv-c))


def gyro_model_praesepe(log10_age, bprp):
    """
    Predicts log10 rotation period from log10 color and log10 age.

    Args:
        log10_age (float): The (log10) age.
        bprp (float): The G_bp - G_rp color.
    Returns:
        log10_period (float): The period.
    """
    log10_bprp = np.log10(bprp)

    # Hard-code the gyro parameters :-)
    p = [-38.957586198640314, 28.709418579540294, -4.919056437046026,
         0.7161114835620975, -4.716819674578521, 0.6470950862322454,
         -13.558898318835137, 0.9359250478865809]

    if log10_bprp >= .43:
        return np.polyval(p[6:], log10_bprp) + p[5]*log10_age
    elif log10_bprp < -.25:
        return 0
    else:
        return np.polyval(p[:5], log10_bprp) + p[5]*log10_age


# def gyro_model_praesepe(log10_age, bprp):
#     """Predict a rotation period from an age and G_BP - G_RP colour.

#     Given a Gaia G_BP - G_RP colour and an age, predict a rotation period
#     using a model fit to the Praesepe cluster alone.

#     Args:
#         log10_age (float or array): The logarithmic age of a star or stars,
#             log10(age), in years.
#         bprp: (float or array): The Gaia G_BP - G_RP colour of a star or
#             stars.

#     Returns:
#         The rotation period in days.

#     """

#     age_gyr = (10**log10_age)*1e-9
#     log_age_gyr = np.log10(age_gyr)
#     log_c = np.log10(bprp)
#     params = [1.10469903, 0.6183025, -4.452133, 31.02877576, -47.76497323,
#               0.63604919]
#     log_P = params[0] + params[1]*log_c + params[2]*log_c**2 \
#         + params[3]*log_c**3 + params[4]*log_c**4 + params[5]*log_age_gyr
#     return 10**log_P


def age_model(log10_period, bprp):
    """
    Predicts log10 age from log10 color and log10 period.

    Args:
        log10_period (array): The (log10) period array.
        log10_bprp (array): The (log10) G_bp - G_rp color array.
    Returns:
        log10_age (array): The (log10) age  array.
    """
    log10_bprp = np.log10(bprp)

    # Hard-code the gyro parameters :-)
    p = [-38.957586198640314, 28.709418579540294, -4.919056437046026,
        0.7161114835620975, -4.716819674578521, 0.6470950862322454,
        -13.558898318835137, 0.9359250478865809]

    if log10_bprp >= .43:
        # return (log10_period - np.polyval(p[6:], log10_bprp))/p[5]
        return 10.14  # The age of the universe
    elif log10_bprp < -.25:
        return 10.14
    else:
        logage = (log10_period - np.polyval(p[:5], log10_bprp))/p[5]
        return logage


def gyro_model_rossby(log10_age, color, mass, Ro_cutoff=2, rossby=True,
                      model="angus15"):
    """Predict a rotation period from an age, B-V colour and mass.

    Predict a rotation period from an age, B-V color and mass using the Angus
    et al. (2015) gyrochronology model with the van Saders et al. (2016)
    weakened magnetic braking correction.

    Args:
        log10_age (float): The log10_age of a star in years.
        color (float or array): The B-V or G_BP - G_RP color of a star.
        mass (float or array): The mass of a star in Solar masses.
        Ro_cutoff (float, optional): The critical Rossby number after which
            stars retain their rotation period. This is 2.16 in van Saders et
            al. (2016) and 2.08 in van Saders et al. (2018). We adopt a
            default value of 2.16.
        rossby (Optional[bool]): If True (default), the van Saders (2016)
            weakened magnetic braking law will be implemented. If false, the
            Angus et al. (2015) gyrochronology relation will be used
            unmodified.

    Returns:
        The log10(rotation period).
    """

    # Angus et al. (2015) parameters.
    a, b, c, n = [.4, .31, .45, .55]

    if not rossby:  # If Rossby model is switched off
        # Standard gyro model
        if model == "angus15":
            if color < c:
                return 0
            log_P = gyro_model(log10_age, color)
        elif model == "praesepe":
            log_P = gyro_model_praesepe(log10_age, color)
        return log_P

    # Otherwise the Rossby model is switched on.
    # Calculate the maximum theoretical rotation period for this mass.
    pmax = Ro_cutoff * convective_overturn_time(mass)

    # Calculate the age this star reaches pmax, based on its B-V color.
    if model == "angus15":
        if color < .45:
            log10_age_thresh = 10.14  # The age of the Universe
        else:
            age_thresh_myr = (pmax/(a*(color-c)**b))**(1./n)
            log10_age_thresh = np.log10(age_thresh_myr*1e6)
    elif model == "praesepe":
        log10_age_thresh = age_model(np.log10(pmax), color)

    # If star younger than critical age, predict rotation from age and color.
    if color < 0. and model=="praesepe":
        log_P = 0
    elif color < .45 and model=="angus15":
        log_P = 0
    elif np.isnan(color) == True:
        log_P = np.nan

    if log10_age < log10_age_thresh:
        if model == "angus15":
            age_myr = (10**log10_age)*1e-6
            if color-c < 0:
                log_P = 0
            else:
                log_P = n*np.log10(age_myr) + np.log10(a) \
                    + b*np.log10(color-c)
        elif model == "praesepe":
            log_P = gyro_model_praesepe(log10_age, color)

    # If star older than this age, return maximum possible rotation period.
    elif log10_age >= log10_age_thresh:
        log_P = np.log10(pmax)

    return log_P


def calc_bv(mag_pars):
    """Calculate a B-V colour from stellar parameters.

    Calculate B-V colour from stellar parameters [EEP, log10(age, yrs), feh,
    distance (in parsecs) and extinction] using MIST isochrones.

    Args:
        mag_pars (list): A list containing EEP, log10(age) in years,
            metallicity, distance in parsecs and V-band extinction, Av, for a
            star.

    Returns:
        B-V color.

    """

    _, _, _, bands = mist.interp_mag([*mag_pars], ["B", "V"])
    B, V = bands
    return B-V


def calc_bprp(mag_pars):
    """Calculate a G_bp-G_rp colour from stellar parameters.
    Calculate bp-rp colour from stellar parameters [EEP, log10(age, yrs), feh,
    distance (in parsecs) and extinction] using MIST isochrones.
    Args:
        mag_pars (list): A list containing EEP, log10(age) in years,
            metallicity, distance in parsecs and V-band extinction, Av, for a
            star.
    Returns:
        G_bp - G_rp color.
    """
    _, _, _, bands = mist.interp_mag([*mag_pars], ["BP", "RP"])
    bp, rp = bands
    return bp - rp


def lnprior(params):
    """ logarithmic prior on parameters.

    The (natural log) prior on the parameters. Takes EEP, log10(age) in years,
    metallicity (feh), distance in parsecs and V-band extinction (Av).

    Args:
        params (array-like): An array of EEP, age, feh, distance and
            extinction.

    Returns:
        The prior probability for the parameters.

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
    """ The ln-probability function.

    Calculates the logarithmic posterior probability (likelihood times prior)
    of the model given the data.

    Args:
        lnparams (array): The parameter array containing Equivalent
            Evolutionary Point (EEP), age in log10(yrs), metallicity, distance
            in ln(pc) and V-band extinction. [EEP, log10(age [yrs]), [Fe/H],
            ln(distance [pc]), A_v].
        *args:
            The arguments -- mod, period, period_err, color, mass, iso_only
            and gyro_only. mod is the isochrones starmodel object which is set
            up in stardate.py. period, period_err, color and mass are the
            rotation period and rotation period uncertainty (in days), color
            (either B-V or G_BP - G_RP) and mass [M_sun].
            color and mass should both be None unless only gyrochronology is
            being used.

    Returns:
        The log-posterior probability of the model given the data.

    """

    # Transform mass and distance back to linear.
    params = lnparams*1
    params[3] = np.exp(lnparams[3])

    # Unpack the args.
    mod, period, period_err, color, mass, iso_only, gyro_only, rossby, \
        model = args

    # If the prior is -inf, don't even try to calculate the isochronal
    # likelihood.
    lnpr = mod.lnprior(params)
    if not np.isfinite(lnpr):
        return -np.inf, -np.inf

    # Put a prior on EEP
    if params[0] > 800:
        return -np.inf, -np.inf

    # If isochrones only, just return the isochronal lhf.
    if iso_only:
        return mod.lnlike(params) + lnpr, lnpr

    if gyro_only == True:
        assert mass, "You must provide a colour if you "\
            "want to calculate an age using gyrochronology only."

    # If a B-V is not provided, calculate it.
    if gyro_only == False:
        if model == "angus15":
            color = calc_bv(params)
        elif model == "praesepe":
            color = calc_bprp(params)

    # If the color value calculated is nan, return the prior.
    # If it is negative, return the iso likelihood.
    if np.isfinite(color) == False:
        return lnpr, lnpr

    if color < 0:
        return mod.lnlike(params) + lnpr, lnpr

    # Check that the period is a positive, finite number. It doesn't matter
    # too much what the lhf is here, as long as it is constant.
    if not period or not np.isfinite(period) or period <= 0.:
        gyro_lnlike = -.5*((5/(20.))**2) - np.log(20.)

    if mass is None:  # If a mass is not provided, calculate it.
        mass = mist.interp_value([params[0], params[1], params[2]],
                                    ["mass"])

    # Calculate a period using the gyrochronology model
    log10_period_model = gyro_model_rossby(params[1], color, mass,
                                           rossby=rossby, model=model)

    var = (period_err/period + sigma(color, params[0]))**2

    # Calculate the gyrochronology likelihood.
    gyro_lnlike = -.5*((log10_period_model - np.log10(period))**2/var) \
        - .5*np.log(2*np.pi*var)

    if gyro_only:
        return float(gyro_lnlike) + lnpr, lnpr

    prob = mod.lnlike(params) + gyro_lnlike + lnpr

    if not np.isfinite(prob):
        prob = -np.inf

    return float(prob), lnpr


def nll(lnparams, args):
    """ The negative ln-probability function.

    Calculates the negative logarithmic posterior probability (likelihood times
    prior) of the model given the data.

    Args:
        lnparams (array): The parameter array containing Equivalent
            Evolutionary Point (EEP), age in log10(yrs), metallicity, distance
            in ln(pc) and V-band extinction. [EEP, log10(age [yrs]), [Fe/H],
            ln(distance [pc]), A_v].
        *args:
            The arguments -- mod, period, period_err, color, mass, iso_only
            and gyro_only. mod is the isochrones starmodel object which is set
            up in stardate.py. period, period_err, color and mass are the
            rotation period and rotation period uncertainty (in days), B-V
            color and mass [M_sun]. color and mass should both be None unless
            only gyrochronology is being used.

    Returns:
        The negative log-posterior probability of the model given the data.

    """
    lp, prior = lnprob(lnparams, *args)
    return -lp


def convective_overturn_time(*args):
    """Estimate the convective overturn time.

    Estimate the convective overturn time using equation 11 in Wright et al.
    (2011): https://arxiv.org/abs/1109.4634
    log tau = 1.16 - 1.49log(M/M⊙) - 0.54log^2(M/M⊙)

    Args:
        args: EITHER mass (float): Mass in Solar units OR eep (float):
            The Equivalent evolutionary point of a star (355 for the Sun),
            age (float): The age of a star in log_10(years) and feh (float):
            the metallicity of a star.

    Returns:
        The convective overturn time in days.

    """

    if len(args) > 1:
        # Convert eep, age and feh to mass (mass will be in Solar mass units)
        eep, age, feh = args
        M = mist.interp_value([eep, age, feh], ["mass"])
    else:
        M = args[0]

    log_tau = 1.16 - 1.49*np.log10(M) - .54*(np.log10(M))**2
    return 10**log_tau


def sigmoid(k, x0, L, x):
    """
    Computes a sigmoid function.
    Args:
        k (float): The logistic growth rate (steepness).
        x0 (float): The location of 1/2 max.
        L (float): The maximum value.
        x, (array): The x-array.
    Returns:
        y (array): The logistic function.
    """
    return L/(np.exp(-k*(x - x0)) + 1)


def sigma(color, eep, model="angus15"):
    """
    The standard deviation of the rotation period distribution.
    Currently comprised of two three logistic functions that 'blow up' the
    variance at hot colours, cool colours and large EEPs. The FGK dwarf part
    of the model has zero variance.
    Args:
        color (float or array): The B-V or G_BP - G_RP colour.
        eep (float or array): The equivalent evolutionary point.
    """
    kcool, khot, keep = 100, 100, .2
    Lcool, Lhot, Leep = .5, .5, .5
    x0eep = 454

    if model == "angus15":
        x0cool, x0hot = 1.4, .45
        sigma_color = sigmoid(kcool, x0cool, Lcool, color) \
            + sigmoid(khot, -x0hot, Lhot, -color)

    elif model == "praesepe":
        x0cool, x0hot = .4, .25
        if color > 0:
            sigma_color = sigmoid(kcool, x0cool, Lcool, np.log10(color)) \
                + sigmoid(khot, x0hot, Lhot, -np.log10(color))
        else:
            sigma_color = .5

    sigma_eep = sigma_eep = sigmoid(keep, x0eep, Leep, eep)
    return sigma_color + sigma_eep


def calc_rossby_number(prot, mass):
    """
    Calculate the Rossby number of a star.
    Args:
        prot (float or array): The rotation period in days.
        mass (float or array): The mass in Solar masses.
    Returns:
        Ro (float or array): The Rossby number.
    """
    return prot/convective_overturn_time(mass)
