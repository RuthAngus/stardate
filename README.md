Star date
===============

CODE:
====

Most recent:

NGC6811_19.ipynb: assembling data for the Kepler clusters.

Praesepe_exploration.ipynb: consolidating cluster data and fitting linear
model.

Cluster_results_notebook.ipynb: running code on clusters.

Hyades.ipynb: attempting to consolidate Hyades data.

chronology.py: contains the general functions that perform the age inference.

Results_notebook.ipynb: a walk through of all the tests described in the
manuscript.

Results_plots.ipynb: creating all the plots in the paper


age_inference.ipynb: infer ages using isochrones and rotation periods.

Crossmatching_and_munging.ipynb: Crossmatch targets with Gaia, CKS, Mcquillan
catalogues, etc.

priors.py: contains priors over parameters.

production_run.ipynb: running on the Sanders sample.

Simulate_data.ipynb: simulate some data to test the model on.

test_on_simulations: a more generalized version of production_run, designed to
run on simulated data.

simulated_data.csv: produced by Simulate_data.ipynb.
combined_data.csv: produced by production_run.ipynb. Resaving data with more
generic column names.


DATA:
=====
hyades.csv: a csv file containing Douglas rotation periods and Gaia data for
Hyads.

Douglas_hyades.csv: a file containing Stephanie Douglas' rotation periods for
the Hyades

praesepe-result.csv: the Gaia DR2 catalogue of praesepe stars.
praesepe_no_outliers.csv: the Gaia DR2 - rotation period crossmatched
catalogue of praesepe stars with outliers removed.
