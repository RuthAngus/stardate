.. stardate documentation master file, created by
   sphinx-quickstart on Sat Nov  3 16:17:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Code for stellar age inference
====================================

stardate combines isochrone fitting with gyrochronology to provide precise
stellar ages.

Example useage
============
::

    import stardate as sd

    iso_params = {"G": (.82, 10),   # Gaia G magnitude with uncertainty
                  "bp": (.4, .01),  # Gaia G_BP with uncertainty
                  "rp": (.4, .01),  # Gaia G_RP with uncertainty
                  "parallax": (10, .01)}  # in milliarcseconds

    prot, prot_err = 26, 1
    star = sd(iso_params, prot, prot_err)
    sampler = star.fit()

    print("stellar age = ", star.age[0], "+", star.age[2],
          "-", star.age[1])

    >> stellar age = 4.5 + 2.1 - 1.3

.. Contents:

.. toctree::
   :maxdepth: 2



.. Indices and tables

.. * :ref:`genindex`
.. * :ref:`modindex`
* :ref:`search`

