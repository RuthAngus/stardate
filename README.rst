.. stardate documentation master file, created by
   sphinx-quickstart on Sat Nov  3 16:17:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

stardate
====================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2712419.svg
   :target: https://doi.org/10.5281/zenodo.2712419

.. image:: http://joss.theoj.org/papers/ee2bbcd6b8fd88492d60f2fe77f4fcdd/status.svg
   :target: http://joss.theoj.org/papers/ee2bbcd6b8fd88492d60f2fe77f4fcdd

Checkout `the documentation <https://stardate.readthedocs.io/en/latest/>`_.

*stardate* is a tool for measuring precise stellar ages.
it combines isochrone fitting with gyrochronology (rotation-based ages) to
increase the precision of stellar ages on the main sequence.
the best possible ages provided by *stardate* will be for stars with rotation
periods, although ages can be predicted for stars without rotation periods
too.
if you don't have rotation periods for any of your stars, you might consider
using `isochrones <https://github.com/timothydmorton/isochrones>`_ as
*stardate* is simply an extension to *isochrones* that incorporates
gyrochronology.
*stardate* reverts back to *isochrones* when no rotation period is provided.

If you would like to contribute to this project, feel free to raise issues or
submit pull requests from the github repo.

Installation
============

.. code-block:: bash
    git clone https://github.com/RuthAngus/stardate.git
    cd stardate
    python setup.py install

Dependencies
------------

The dependencies of *stardate* are
`NumPy <http://www.numpy.org/>`_,
`pandas <https://pandas.pydata.org/>`_,
`h5py <https://www.h5py.org/>`_,
`numba <http://numba.pydata.org/>`_,
`tqdm <https://tqdm.github.io/>`_ and
`isochrones <https://github.com/timothydmorton/isochrones>`_.

The first five of these can be installed using conda or pip:

.. code-block:: bash

    conda install numpy pandas h5py numba tqdm

or

.. code-block:: bash

    pip install numpy pandas h5py numba tqdm

You'll also need to download isochrones:

.. code-block:: bash

    git clone https://github.com/timothydmorton/isochrones
    cd isochrones
    python setup.py install

You can check out the
`isochrones <https://isochrones.readthedocs.io/en/latest/index.html>`_
documentation if you run into difficulties installing that.

Example usage
-------------
::

    import stardate as sd

    # Create a dictionary of observables
    iso_params = {"teff": (4386, 50),     # Teff with uncertainty.
                  "logg": (4.66, .05),    # logg with uncertainty.
                  "feh": (0.0, .02),      # Metallicity with uncertainty.
                  "parallax": (1.48, .1),  # Parallax in milliarcseconds.
                  "maxAV": .1}            # Maximum extinction

    prot, prot_err = 29, 3

    # Set up the star object.
    star = sd.Star(iso_params, prot=prot, prot_err=prot_err)  # Here's where you add a rotation period

    # Run the MCMC
    star.fit(max_n=1000)

    # max_n is the maximum number of MCMC samples. I recommend setting this
    # much higher when running for real, or using the default value of 100000.

    # Print the median age with the 16th and 84th percentile uncertainties.
    age, errp, errm, samples = star.age_results()
    print("stellar age = {0:.2f} + {1:.2f} + {2:.2f}".format(age, errp, errm))

    >> stellar age = 2.97 + 0.60 + 0.55

If you want to just use a simple gyrochronology model without running MCMC,
you can predict a stellar age from a rotation period like this:

::

    import numpy as np
    from stardate.lhf import age_model

    bprp = .82  # Gaia BP - RP color.
    log10_period = np.log10(26)
    log10_age_yrs = age_model(log10_period, bprp)
    print((10**log10_age_yrs)*1e-9, "Gyr")
    >> 4.565055357152765 Gyr

Or a rotation period from an age like this:

::

    from stardate.lhf import gyro_model_praesepe

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
