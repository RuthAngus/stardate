.. stardate documentation master file, created by
   sphinx-quickstart on Sat Nov  3 16:17:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Code for stellar age inference
====================================

stardate combines isochrone fitting with gyrochronology to provide precise
stellar ages.

Installation
============
>> git clone https://github.com/RuthAngus/stardate.git
>> cd stardate
>> python setup.py install


Example useage
============
::

    import stardate as sd

    iso_params = {"G": (.82, 10),   # Gaia G magnitude with uncertainty
                  "bp": (.4, .01),  # Gaia G_BP with uncertainty
                  "rp": (.4, .01),  # Gaia G_RP with uncertainty
                  "parallax": (10, .01)}  # in milliarcseconds

    prot, prot_err = 26, 1
    star = sd.star(iso_params, prot, prot_err)
    sampler = star.fit()

    print("stellar age = ", star.age[0], "+", star.age[2],
          "-", star.age[1])

    >> stellar age = 4.5 + 2.1 - 1.3

    # Accessing posteriors samples over age, mass, metallicity, distance and extinction

    median_mass, lower_mass_err, upper_mass_err, mass_samples = star.mass
    median_age, lower_age_err, upper_age_err, age_samples = star.age
    median_feh, lower_feh_err, upper_feh_err, feh_samples = star.feh
    median_distance, lower_distance_err, upper_distance_err, distance_samples = star.distance
    median_Av, lower_Av_err, upper_Av_err, Av_samples = star.Av
