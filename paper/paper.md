---
title: "stardate:  Combining dating methods for better stellar ages"
tags:
  - example
  - tags
  - for the paper
authors:
 - name: Ruth Angus
   orcid: 0000-0003-4540-5661
   affiliation: "1, 2"
 - name: Timothy D. Morton
   affiliation: "3, 2"
 - name: Daniel Foreman-Mackey
   orcid: 0000-0002-9328-5652
   affiliation: "2"
affiliations:
 - name: Department of Astrophysics, American Museum of Natural History, New York, NY, 10024, USA
   index: 1
 - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY, 10010, USA
   index: 2
 - name: Department of Astronomy, University of Florida, Gainesville, FL, 32611, USA
   index: 3
date: 30 April 2019
bibliography: references.bib
---

# Summary

Age is the most difficult fundamental stellar property to measure,
particularly for stars on the main sequence.
These stars change slowly in brightness and temperature, so measuring their
ages via placement on the Hertzsprung-Russell or color-magnitude diagram can
be imprecise.
``stardate`` combines alternative dating methods with isochrone placement to
infer more precise and accurate ages than any one method alone.
Users provide observable stellar properties that place stars on a
Hertzsprung-Russell or color-magnitude diagram, such as apparent magnitudes,
parallax, plus spectroscopic parameters and asteroseismic parameters (if
available).
They can also provide other information relating to stellar age, such as a
rotation period.
Based on these observables, ``stardate`` combines the different dating methods
to estimate posterior probability density functions (PDFs) over stellar age,
as well as other parameters such as distance, extinction, metallicity and mass
or evolutionary stage.
``stardate`` uses the ``emcee`` Python package [@emcee] to perform Markov
Chain Monte Carlo sampling in order to estimate the posterior PDFs of the
stellar parameters.
The paper describing the method in more detail is Angus et al. (submitted).

``stardate`` is built on top of the ``isochrones`` Python package
[@isochrones] and uses the MIST stellar evolution models [@dotter2016],
[@choi2016].
Development of ``stardate`` happens on GitHub and any issues can be raised
there.

# References
