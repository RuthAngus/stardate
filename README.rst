.. stardate documentation master file, created by
   sphinx-quickstart on Sat Nov  3 16:17:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

stardate
====================================

.. <p>
.. <a href="https://github.com/ruthangus/stardate">
.. <img src="https://zenodo.org/badge/134314438.svg"></a>
.. </p>
<p>
<a href="https://github.com/dfm/exoplanet">
<img src="https://img.shields.io/badge/GitHub-dfm%2Fexoplanet-blue.svg?style=flat"></a>
</p>

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

.. Installation
.. ============

.. .. code-block:: bash
..     git clone https://github.com/RuthAngus/stardate.git
..     cd stardate
..     python setup.py install

.. You'll also need to download isochrones and switch to the eep branch:

.. .. code-block:: bash
..     git clone https://github.com/timothydmorton/isochrones
..     cd isochrones
..     git checkout eep
..     python setup.py install

.. In order to get started you can create a dictionary containing the observables
.. you have for your star.
.. These could be atmospheric parameters (like those shown in the example below
.. for the Sun), or just photometric colors, like those from *2MASS*, *SDSS* or
.. *Gaia*.
.. If you have a parallax, asteroseismic parameters, or an idea of the
.. maximum V-band extinction you should throw those in too.
.. Set up the star object and :func:`chronology.star.fit` will run Markov Chain
.. Monte Carlo (using *emcee*) in order to infer a Bayesian age for your star.

.. Example usage
.. =============
.. ::

..     import stardate as sd

..     # Create a dictionary of observables
..     iso_params = {"teff": (5777, 10),     # Teff with uncertainty.
..                   "logg": (4.44, .05),    # logg with uncertainty.
..                   "feh": (0., .001),      # Metallicity with uncertainty.
..                   "parallax": (1., .01),  # Parallax in milliarcseconds.
..                   "maxAV": .1}            # Maximum extinction

..     prot, prot_err = 26, 1

..     # Set up the star object.
..     star = sd.star(iso_params, prot, prot_err)  # Here's where you add a rotation period

..     # Run the MCMC
..     star.fit()

..     # Print the median age with the 16th and 84th percentile uncertainties.
..     print("stellar age = {0} + {1} + {2}".format(star.age[0], star.age[2], star.age[1])

..     >> stellar age = 4.5 + 2.1 - 1.3
