import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import tqdm
from stardate.lhf import run_mcmc#, make_plots
from isochrones import StarModel
import pandas as pd
import emcee
import corner

from isochrones.mist import MIST_Isochrone
mist = MIST_Isochrone()

plotpar = {'axes.labelsize': 18,
           'font.size': 18,
           'legend.fontsize': 18,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


class star(object):

    def __init__(self, iso_params, prot, prot_err, savedir=".",
                 filename="samples", bv=None, mass=None):
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
            (optional) The name of the directory where the samples will be
            saved. Default is the current working directory.
        filename: str
            (optional) The name of the h5 file which the posterior samples
            will be saved in.
        """

        self.iso_params = iso_params
        print(type(iso_params))
        self.prot = prot
        self.prot_err = prot_err
        self.savedir = savedir
        self.filename = filename
        self.bv = bv
        self.mass = mass

    def fit(self, inits=[355, np.log10(4.56*1e9), 0., 1000., .01],
            nwalkers=24, max_n=100000, thin_by=100, iso_only=False,
            rossby=True):
        """
        params
        ------
        inits: (list, optional)
            A list of default initial values to use for eep, age (in
            log10[yrs]), feh, distance (in pc) and Av, if alternatives are not
            provided.
        nwalkers: (int, optional)
            The number of walkers to use with emcee.
        max_n: (int, optional)
            The maximum number of samples to obtain.
        iso_only: (boolean, optional)
            If true only the isochronal likelihood function will be used.
        """

        p_init = [inits[0], inits[1], inits[2], np.log(inits[3]), inits[4]]

        np.random.seed(42)

        # Create the directory if it doesn't already exist.
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        # Set up the backend
        # Don't forget to clear it in case the file already exists
        filename = "{0}/{1}_samples.h5".format(self.savedir, self.filename)
        backend = emcee.backends.HDFBackend(filename)
        nwalkers, ndim = 24, 5
        backend.reset(nwalkers, ndim)

        # Set up the StarModel object needed to calculate the likelihood.
        mod = StarModel(mist, **self.iso_params)  # StarModel isochrones obj

        args = [mod, self.prot, self.prot_err, self.bv, self.mass, iso_only, rossby]  # lnprob arguments

        # Run the MCMC
        sampler = run_mcmc(self.iso_params, args, p_init, backend, ndim=ndim,
                           nwalkers=nwalkers, max_n=max_n, thin_by=thin_by)

        self.sampler = sampler
        self.samples = sampler.flatchain
        return sampler

    def age(self, burnin=10000):
        """
        params
        ------
        burnin: int
            The number of samples to cut off at the beginning of the MCMC
            when calculating the posterior percentiles.
        Returns the median age and lower and upper uncertainties.
        Age is log10(Age/yrs).
        """
        samples = self.sampler.chain[:, burnin:, 1]
        nwalkers, nsteps = np.shape(samples)
        samps = np.reshape(samples, (nwalkers*nsteps))
        a = np.median(samps)
        errp = np.percentile(samps, 84) - a
        errm = a - np.percentile(samps, 16)
        return a, errm, errp, samps


    def eep(self, burnin=10000):
        """
        params
        ------
        burnin: int
            The number of samples to cut off at the beginning of the MCMC
            when calculating the posterior percentiles.
        Returns the median mass and lower and upper uncertainties in units of
        solar mass.
        """
        samples = self.sampler.chain[:, burnin:, 0]
        nwalkers, nsteps = np.shape(samples)
        samps = np.reshape(samples, (nwalkers*nsteps))
        e = np.median(samps)
        errp = np.percentile(samps, 84) - e
        errm = e - np.percentile(samps, 16)
        return e, errm, errp, samps


    def mass(self, burnin=10000):
        """
        params
        ------
        burnin: int
            The number of samples to cut off at the beginning of the MCMC
            when calculating the posterior percentiles.
        Returns the median mass and lower and upper uncertainties in units of
        solar mass.
        """
        samples = self.sampler.chain[:, burnin:, :]
        nwalkers, nsteps, ndim = np.shape(samples)
        samps = np.reshape(samples, (nwalkers*nsteps, ndim))
        msamps = mist.mass(samps[:, 0], samps[:, 1], samps[:, 2])
        m = np.median(msamps)
        errp = np.percentile(msamps, 84) - m
        errm = m - np.percentile(esamps, 16)
        return m, errm, errp, msamps


    def feh(self, burnin=10000):
        """
        params
        ------
        burnin: int
            The number of samples to cut off at the beginning of the MCMC
            when calculating the posterior percentiles.
        Returns the median metallicity and lower and upper uncertainties.
        """
        samples = self.sampler.chain[:, burnin:, 2]
        nwalkers, nsteps = np.shape(samples)
        samps = np.reshape(samples, (nwalkers*nsteps))
        f = np.median(samps)
        errp = np.percentile(samps, 84) - f
        errm = f - np.percentile(samps, 16)
        return f, errm, errp, samps


    def distance(self, burnin=10000):
        """
        params
        ------
        burnin: int
            The number of samples to cut off at the beginning of the MCMC
            when calculating the posterior percentiles.
        Returns the median distance and lower and upper uncertainties in
        parsecs.
        """
        samples = self.sampler.chain[:, burnin:, 3]
        nwalkers, nsteps = np.shape(samples)
        samps = np.reshape(samples, (nwalkers*nsteps))
        d = np.median(samps)
        errp = np.percentile(samps, 84) - d
        errm = d - np.percentile(samps, 16)
        return d, errm, errp, samps


    def Av(self, burnin=10000):
        """
        params
        ------
        burnin: int
            The number of samples to cut off at the beginning of the MCMC
            when calculating the posterior percentiles.
        Returns the median distance and lower and upper uncertainties in
        parsecs.
        """
        samples = self.sampler.chain[:, burnin:, 4]
        nwalkers, nsteps = np.shape(samples)
        samps = np.reshape(samples, (nwalkers*nsteps))
        a_v = np.median(samps)
        errp = np.percentile(samps, 84) - a_v
        errm = a_v - np.percentile(samps, 16)
        return a_v, errm, errp, samps


    # def make_plots(self, truths=[None, None, None, None, None], burnin=10000):
    #     """
    #     params
    #     ------
    #     truths: list
    #         A list of true values to give to corner.py that will be plotted
    #         in corner plots. If an entry is "None", no line will be plotted.
    #         Default = [None, None, None, None, None]
    #     burnin: int
    #         The number of burn in samples at the beginning of the MCMC to
    #         throw away. The default is 100000.
    #     """

    #     nwalkers, nsteps, ndim = np.shape(self.sampler.chain)
    #     print("nsteps = ", nsteps, "burnin = ", burnin)
    #     assert burnin < nsteps, "The number of burn in samples to throw" \
    #         "away can't exceed the number of steps."

    #     # samples = self.sampler.flatchain
    #     samples = np.reshape(self.sampler.chain[:, burnin:, :],
    #                          (nwalkers*(nsteps - burnin), ndim))

    #     print("Plotting age posterior")
    #     age_gyr = (10**samples[:, 1])*1e-9
    #     plt.hist(age_gyr)
    #     plt.xlabel("Age [Gyr]")
    #     med, std = np.median(age_gyr), np.std(age_gyr)
    #     if truths[2]:
    #         plt.axvline(10**(truths[2])*1e-9, color="tab:orange",
    #                     label="$\mathrm{True~age~[Gyr]}$")
    #     plt.axvline(med, color="k", label="$\mathrm{Median~age~[Gyr]}$")
    #     plt.axvline(med - std, color="k", linestyle="--")
    #     plt.axvline(med + std, color="k", linestyle="--")
    #     plt.savefig("{0}/{1}_marginal_age".format(self.savedir,
    #                                               str(self.suffix).zfill(4)))
    #     plt.close()

    #     print("Plotting production chains...")
    #     plt.figure(figsize=(16, 9))
    #     for j in range(ndim):
    #         plt.subplot(ndim, 1, j+1)
    #         plt.plot(self.sampler.chain[:, :, j].T, "k", alpha=.1)
    #         plt.axhline(truths[j+1])
    #     plt.savefig("{0}/{1}_chains".format(self.savedir,
    #                                         str(self.suffix).zfill(4)))
    #     plt.close()

    #     print("Making corner plot...")
    #     labels = ["$\mathrm{Mass~}[M_\odot]$", "$\mathrm{EEP}$",
    #               "$\log_{10}(\mathrm{Age~[yr]})$",
    #               "$\mathrm{[Fe/H]}$",
    #               "$\ln(\mathrm{Distance~[Kpc])}$",
    #               "$A_v$"]
    #     mass_samples = np.zeros((nwalkers*(nsteps-burnin), ndim+1))
    #     mass_samples[:, 1:] = samples[:, :]
    #     mass_samples[:, 0] = mist.mass(samples[:, 0], samples[:, 1],
    #                                    samples[:, 2])
    #     corner.corner(mass_samples[:, :], labels=labels, truths=truths);
    #     plt.savefig("{0}/{1}_corner".format(self.savedir,
    #                                         str(self.suffix).zfill(4)))
    #     plt.close()

    #     # # Make mass histogram
    #     # samples = self.sampler.flatchain
    #     # mass_samps = mist.mass(samples[:, 0], samples[:, 1], samples[:, 2])
    #     # plt.hist(mass_samps, 50);
    #     # if truths[0]:
    #     #         plt.axvline(mist.mass(truths[0], truths[1], truths[2]),
    #     #                               color="tab:orange",
    #     #                     label="$\mathrm{True~mass~}[M_\odot]$")
    #     # plt.savefig("{0}/{1}_marginal_mass".format(self.savedir,
    #     #                                            str(self.suffix).zfill(4)))
    #     # plt.close()
