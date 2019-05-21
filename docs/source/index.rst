.. stardate documentation master file, created by
   sphinx-quickstart on Sat Nov  3 16:17:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

stardate
====================================

*stardate* is a tool for measuring precise stellar ages.
It combines isochrone fitting with gyrochronology (rotation-based age
inference) to increase the precision of stellar ages on the main sequence.
The best possible ages provided by *stardate* will be for stars with rotation
periods, although ages can be predicted for stars without rotation periods
too.
If you don't have rotation periods for any of your stars, you might consider
using `isochrones.py <https://github.com/timothydmorton/isochrones>`_ as
*stardate* is simply an extension to *isochrones* that incorporates
gyrochronology.
*stardate* reverts back to *isochrones* when no rotation period is provided.

In order to get started you can create a dictionary containing the observables
you have for your star.
These could be atmospheric parameters (like those shown in the example below
for the Sun), or just photometric colors, like those from *2MASS*, *SDSS* or
*Gaia*.
If you have a parallax, asteroseismic parameters, or an idea of the
maximum V-band extinction you should throw those in too.
Set up the star object and :func:`stardate.Star.fit` will run Markov Chain
Monte Carlo (using *emcee*) in order to infer a Bayesian age for your star.

Note -- if you are running the example below for the first time, the
isochrones will be downloaded and this could take a while.
This will only happen once though!

Example usage
-------------
::

    import stardate as sd

    # Create a dictionary of observables
    iso_params = {"teff": (5777, 10),     # Teff with uncertainty.
                  "logg": (4.44, .05),    # logg with uncertainty.
                  "feh": (0., .001),      # Metallicity with uncertainty.
                  "parallax": (1., .01),  # Parallax in milliarcseconds.
                  "B": (15.48, 0.02),     # You must provide at least one magnitude.
                  "maxAV": .1}            # Maximum extinction

    prot, prot_err = 26, 1

    # Set up the star object.
    star = sd.Star(iso_params, prot=prot, prot_err=prot_err)  # Here's where you add a rotation period

    # Run the MCMC
    star.fit()

    # Print the median age with the 16th and 84th percentile uncertainties.
    age, errp, errm, samples = star.age_results()
    print("stellar age = {0} + {1} + {2}".format(age, errp, errm))

    >> stellar age = 4.5 + 2.1 - 1.3

If you want to just use a simple gyrochronology model without running MCMC,
you can predict a stellar age from a rotation period like this:

::

    from stardate.lhf import age_model

    bprp = .82  # Gaia BP - RP color.
    log10_period = np.log10(26)
    log10_age_yrs = age_model(log10_period, bprp)
    print((10**log10_age_yrs)*1e-9, "Gyr")
    >> 4.565055357152765 Gyr

Or a rotation period from an age like this:

::

    from stardate.lhf import gyro_model_prasepe

    bprp = .82  # Gaia BP - RP color.
    log10_age_yrs = np.log10(4.56*1e9)
    log10_period = gyro_model_praesepe(log10_age_yrs, bprp)
    print(10**log10_period, "days")
    >> 25.98136488222407 days

BUT be aware that these simple relations are only applicable to FGK and early
M dwarfs on the main sequence, older than a few hundred Myrs.
If you're not sure if gyrochronology is applicable to your star, want the best
age possible, or would like proper uncertainty estimates, I recommend using
the full MCMC approach.

.. Contents:

User Guide
----------

.. toctree::
   :maxdepth: 2

   user/install
   user/api


Tutorials
---------

.. toctree::
   :maxdepth: 2

   tutorials/Tutorial


License & attribution
---------------------

Copyright 2018, Ruth Angus.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite this package and its dependencies.
You can find more information about how and what to cite in the
:ref:`citation` documentation.

* :ref:`search`

