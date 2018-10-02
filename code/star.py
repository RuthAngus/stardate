import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import tqdm
from chronology import setup, run_mcmc, make_plots
from isochrones import StarModel
import pandas as pd
import emcee
import corner

from isochrones.mist import MIST_Isochrone
mist = MIST_Isochrone()

plotpar = {'axes.labelsize': 25,
           'font.size': 25,
           'legend.fontsize': 25,
           'xtick.labelsize': 25,
           'ytick.labelsize': 25,
           'text.usetex': True}
plt.rcParams.update(plotpar)


class star(object):

    def __init__(self, iso_params, prot, prot_err, savedir=".", suffix="_"):
        """
        params
        -------
        iso_params: dictionary
            A dictionary containing all available photometric and
            spectroscopic parameters for a star, as well as its parallax.
            All parameters should also have associated uncertainties.
            This dictionary should be similar to the standard one created for
            isochrones.py.
        prot: float
            The rotation period of the star in days.
        prot_err: float
            The uncertainty on the stellar rotation period in days.
        savedir: str
            The name of the directory where the samples will be saved.
        suffix: str
            The id or name of the star to use in the filename.
        """

        self.iso_params = iso_params
        self.prot = prot
        self.prot_err = prot_err
        self.savedir = savedir
        self.suffix = suffix

    def fit(self, inits=[1., 9., 0., .5, .01], nwalkers=24, max_n=100000,
            iso_only=False):
        """
        params
        ------
        inits: list
            A list of default initial values to use for mass, age, feh,
            distance and Av, if alternatives are not provided.
        nwalkers: int
            The number of walkers to use with emcee.
        max_n: int
            The maximum number of samples to obtain.
        iso_only: boolean
            If true only the isochronal likelihood function will be used.
        """

        # Set the initial values
        mass_init, age_init, feh_init, distance_init, Av_init = inits
        eep_init = mist.eep_from_mass(mass_init, age_init, feh_init)

        # sample in linear eep, log10(age), linear feh, ln(distance) and
        # linear Av.
        p_init = np.array([eep_init, age_init, feh_init,
                           np.log(distance_init), Av_init])

        np.random.seed(42)

        # Set up the backend
        # Don't forget to clear it in case the file already exists
        filename = "{0}/{1}_samples.h5".format(self.savedir,
                                               str(self.suffix).zfill(4))
        backend = emcee.backends.HDFBackend(filename)
        nwalkers, ndim = 24, 5
        backend.reset(nwalkers, ndim)

        # Set up the StarModel object needed to calculate the likelihood.
        mod = StarModel(mist, **self.iso_params)  # StarModel isochrones obj
        args = [mod, self.prot, self.prot_err, iso_only]  # lnprob arguments

        # Run the MCMC
        sampler = run_mcmc(self.iso_params, args, p_init, backend, ndim=ndim,
                           nwalkers=nwalkers, max_n=max_n)

        self.sampler = sampler
        return sampler


    def make_plots(self, truths=[None, None, None, None, None], burnin=10000):
        """
        params
        ------
        truths: list
            A list of true values to give to corner.py that will be plotted
            in corner plots. If an entry is "None", no line will be plotted.
            Default = [None, None, None, None, None]
        burnin: int
            The number of burn in samples at the beginning of the MCMC to
            throw away. The default is 100000.
        """

        nwalkers, nsteps, ndim = np.shape(self.sampler.chain)
        print("nsteps = ", nsteps, "burnin = ", burnin)
        assert burnin < nsteps, "The number of burn in samples to throw" \
            "away can't exceed the number of steps."

        samples = self.sampler.flatchain

        print("Plotting age posterior")
        age_gyr = (10**samples[burnin:, 1])*1e-9
        plt.hist(age_gyr)
        plt.xlabel("Age [Gyr]")
        med, std = np.median(age_gyr), np.std(age_gyr)
        if truths[1]:
            plt.axvline(10**(truths[1])*1e-9, color="tab:orange",
                        label="$\mathrm{True~age~[Gyr]}$")
        plt.axvline(med, color="k", label="$\mathrm{Median~age~[Gyr]}$")
        plt.axvline(med - std, color="k", linestyle="--")
        plt.axvline(med + std, color="k", linestyle="--")
        plt.savefig("{0}/{1}_marginal_age".format(self.savedir,
                                                  str(self.suffix).zfill(4)))
        plt.close()

        print("Plotting production chains...")
        plt.figure(figsize=(16, 9))
        for j in range(ndim):
            plt.subplot(ndim, 1, j+1)
            plt.plot(self.sampler.chain[:, burnin:, j].T, "k", alpha=.1)
        plt.savefig("{0}/{1}_chains".format(self.savedir,
                                            str(self.suffix).zfill(4)))
        plt.close()

        print("Making corner plot...")
        labels = ["$\mathrm{EEP}$",
                  "$\log_{10}(\mathrm{Age~[yr]})$",
                  "$\mathrm{[Fe/H]}$",
                  "$\ln(\mathrm{Distance~[Kpc])}$",
                  "$A_v$"]
        corner.corner(samples[burnin:, :], labels=labels, truths=truths);
        plt.savefig("{0}/{1}_corner".format(self.savedir,
                                            str(self.suffix).zfill(4)))
        plt.close()

        # Make mass histogram
        samples = self.sampler.flatchain
        mass_samps = mist.mass(samples[:, 0], samples[:, 1], samples[:, 2])
        plt.hist(mass_samps, 50);
        if truths[0]:
                plt.axvline(truths[0], color="tab:orange",
                            label="$\mathrm{True~mass~}[M_\odot]$")
        plt.savefig("{0}/{1}_marginal_mass".format(self.savedir,
                                                   str(self.suffix).zfill(4)))
        plt.close()
