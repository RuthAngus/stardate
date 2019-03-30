import os
import numpy as np
import pandas as pd
import h5py
import tqdm
from .lhf import lnprob, nll
from isochrones import StarModel, get_ichrone
import pandas as pd
import emcee
import scipy.optimize as spo

from isochrones.mist import MIST_Isochrone
bands = ["B", "V", "J", "H", "K"]
# mist = MIST_Isochrone(bands)
mist = get_ichrone("mist", bands=bands)


class Star(object):
    """The star object.

    Creates the star object which will be set up with stellar parameters and
    instructions for saving the MCMC results.

    Args:
        iso_params (dict): A dictionary containing all available photometric
            and spectroscopic parameters for a star, as well as its parallax.
            All parameters should also have associated uncertainties.
            This dictionary should be similar to the standard one created for
            isochrones.py.
        prot (float): The rotation period of the star in days.
        prot_err (float): The uncertainty on the stellar rotation period in
            days.
        bv (Optional[float]): B-V color. In order to infer an age with only
            gyrochronology, a B-V color must be provided. Only include this if
            you want to use gyrochronology only!
        mass (Optional[float]): In order to infer an age with only
            gyrochronology, a mass must be provided (units of Solar masses).
            Only include this if you want to use gyrochronology only!
        savedir (Optional[str]): The name of the directory where the samples
            will be saved. Default is the current working directory.
        filename (Optional[str]): The name of the h5 file which the posterior
            samples will be saved in.

    """

    def __init__(self, iso_params, prot=None, prot_err=None, bv=None,
                 mass=None, savedir=".", filename="samples"):

        self.iso_params = iso_params
        self.prot = prot
        self.prot_err = prot_err
        self.savedir = savedir
        self.filename = filename
        self.bv = bv
        self.mass = mass

    def fit(self, inits=[355, 9.659, 0., 1000., .01], nwalkers=24,
            max_n=100000, thin_by=100, burnin=0, iso_only=False,
            gyro_only=False):
        """Run MCMC on a star.

        Explore the posterior probability density function of the stellar
        parameters using MCMC (via emcee).

        Args:
            inits (Optional[array-like]): A list of initial values to use for
            EEP, age (in log10[yrs]), feh, distance (in pc) and Av. The
            defaults are Solar values at 1000 pc with .01 extinction.
            nwalkers (Optional[int]): The number of walkers to use with emcee.
            The default is 24.
            max_n (Optional[int]): The maximum number of samples to obtain
                (although not necessarily to save -- see thin_by). The default
                is 100000.
            thin_by (Optional[int]): Only one in every thin_by samples will be
                saved. The default is 100. Set = 1 to save every sample (note
                this substantially slows down the MCMC process because of the
                additional I/O time.
            burnin (Optional[int]): The number of SAVED samples to throw away
                when accessing the results. This number cannot exceed the
                number of saved samples (which is max_n/thin_by). Default = 0.
            iso_only (Optional[bool]): If true only the isochronal likelihood
                function will be used.
            gyro_only (Optional[bool]): If true only the gyrochronal
                likelihood function will be used. Cannot be true if iso_only
                is true.

        """

        self.max_n = max_n
        self.nwalkers = nwalkers
        self.thin_by = thin_by

        if iso_only:
            assert gyro_only == False, "You cannot set both iso_only and "\
                "gyro_only to be True."

        if gyro_only:
            assert self.mass, "If gyro_only is set to True, you must " \
                "provide a B-V colour and a mass."

        if burnin > max_n/thin_by:
            burnin = int(max_n/thin_by/3)
            print("Automatically setting burn in to {}".format(burnin))

        p_init = [inits[0], inits[1], inits[2], np.log(inits[3]), inits[4]]
        # self.p_init = p_init

        np.random.seed(42)

        # Create the directory if it doesn't already exist.
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        # Set up the backend
        # Don't forget to clear it in case the file already exists
        fn = "{0}/{1}.h5".format(self.savedir, self.filename)
        backend = emcee.backends.HDFBackend(fn)
        nwalkers, ndim = 24, 5
        backend.reset(nwalkers, ndim)
        self.backend = backend

        # Set up the StarModel object needed to calculate the likelihood.
        mod = StarModel(mist, **self.iso_params)  # StarModel isochrones obj

        # lnprob arguments
        args = [mod, self.prot, self.prot_err, self.bv, self.mass, iso_only,
                gyro_only]
        self.args = args

        # optimize
        results = spo.minimize(-nll, p_init, args=args)
        self.p_init = results.x

        # Run the MCMC
        # sampler = run_mcmc(args, p_init, backend, nwalkers=nwalkers,
        #                    max_n=max_n, thin_by=thin_by)
        sampler = self.run_mcmc()
        self.sampler = sampler

    def samples(self, burnin=0):
        """Provides posterior samples.

        Provides the posterior samples and allows the user to specify the
        number of samples to throw away as burn in.

        Args:
            burnin (Optional[int]):
                The number of samples to throw away as burn in. Default = 0.

        Returns:
            samples (array):
                An array containing posterior samples with shape =
                (nwalkers*(nsteps - burnin), ndim).
        """
        nwalkers, nsteps, ndim = np.shape(self.sampler.chain)
        return np.reshape(self.sampler.chain[:, burnin:, :],
                          (nwalkers*(nsteps-burnin), ndim))

    def run_mcmc(self):
        """Runs the MCMC.

        Runs the MCMC using emcee. Saves progress to the file specified as
        <filename.h5> in the <savedir> directory. Samples are saved to this
        file after every <thin_by> samples are obtained. And only one sample
        in every <thin_by> is saved. Sampling continues until either max_n
        samples are obtained or convergence is acheived (this usually takes
        much more than the default 100,000 maximum number of samples.
        Convergence is achieved if both 1) the change in autocorrelation time
        is extremely small (less than 0.01) and 2) if 100 or more independent
        samples have been obtained.

        Returns:
            sampler: the emcee sampler object.
        """

        max_n = self.max_n//self.thin_by

        # p0 = [p_init + np.random.randn(ndim)*1e-4 for k in range(nwalkers)]

        # Broader gaussian for EEP initialization
        ndim = len(self.p_init)  # Should always be 5. Hard code it?
        p0 = np.empty((self.nwalkers, ndim))
        p0[:, 0] = np.random.randn(self.nwalkers)*10 + self.p_init[0]
        p0[:, 1] = np.random.randn(self.nwalkers)*1e-4 + self.p_init[1]
        p0[:, 2] = np.random.randn(self.nwalkers)*1e-4 + self.p_init[2]
        p0[:, 3] = np.random.randn(self.nwalkers)*1e-4 + self.p_init[3]
        p0[:, 4] = np.random.randn(self.nwalkers)*1e-4 + self.p_init[4]
        p0 = list(p0)

        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, lnprob,
                                        args=self.args, backend=self.backend)

        # Copied from https://emcee.readthedocs.io/en/latest/tutorials/monitor/
        # ======================================================================

        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(max_n)

        # This will be useful to testing convergence
        old_tau = np.inf

        # Now we'll sample for up to max_n steps
        for sample in sampler.sample(p0, iterations=max_n,
                                     thin_by=self.thin_by, store=True,
                                     progress=True):
            # Only check convergence every 100 steps
            # if sampler.iteration % 100:
            #     continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            if sampler.iteration > 100:
                tau = sampler.get_autocorr_time(tol=0) * self.thin_by
                autocorr[index] = np.mean(tau)
                index += 1

                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                converged &= np.all(tau) > 1
                # print("100 samples?", np.all(tau * 100 < sampler.iteration))
                # print(tau, tau*100, sampler.iteration)
                # print("Small delta tau?", np.all(np.abs(old_tau - tau) / tau < 0.01))
                # print(np.abs(old_tau - tau))
                if converged:
                    break
                old_tau = tau
        # ======================================================================

        return sampler


    def age_results(self, burnin=0):
        """The age samples.

        The posterior samples for age, optionally with a specified number of
        burn in steps thrown away.

        Args:
            burnin (Optional[int]): The number of samples to throw away.
                Default is 0.

        Returns:
            The median age, its 16th and 84th percentile lower and upper
            uncertainties and the age samples. Age is log10(Age/yrs).

        """

        nwalkers, nsteps, ndim = np.shape(self.sampler.chain)
        assert nsteps > burnin, "The number of burn in samples to throw "\
            "away ({0}) cannot exceed the number of steps that were saved "\
            "({1}). Try setting the burnin keyword argument.".format(burnin,
                                                                     nsteps)
        samples = self.sampler.chain[:, burnin:, 1]
        samps = np.reshape(samples, (nwalkers*(nsteps - burnin)))
        a = np.median(samps)
        errp = np.percentile(samps, 84) - a
        errm = a - np.percentile(samps, 16)
        return a, errm, errp, samps


    def eep_results(self, burnin=0):
        """The EEP samples.

        The posterior samples for Equivalent Evolutionary Point, optionally
        with a specified number of burn in steps thrown away.

        Args:
            burnin (Optional[int]): The number of samples to throw away.
                Default is 0.

        Returns:
            The median EEP, its 16th and 84th percentile lower and upper
            uncertainties and the EEP samples.

        """

        nwalkers, nsteps, ndim = np.shape(self.sampler.chain)
        assert nsteps > burnin, "The number of burn in samples to throw "\
            "away ({0}) cannot exceed the number of steps that were saved "\
            "({1}). Try setting the burnin keyword argument.".format(burnin,
                                                                     nsteps)
        samples = self.sampler.chain[:, burnin:, 0]
        samps = np.reshape(samples, (nwalkers*(nsteps - burnin)))
        e = np.median(samps)
        errp = np.percentile(samps, 84) - e
        errm = e - np.percentile(samps, 16)
        return e, errm, errp, samps


    def mass_results(self, burnin=0):
        """The mass samples.

        The posterior samples for mass, calculated from the EEP, age and feh
        samples, optionally with a specified number of burn in steps thrown
        away.

        Args:
            burnin (Optional[int]): The number of samples to throw away.
                Default is 0.

        Returns:
            The median mass, its 16th and 84th percentile lower and upper
            uncertainties and the mass 'samples' in units of Solar masses.

        """

        nwalkers, nsteps, ndim = np.shape(self.sampler.chain)
        assert nsteps > burnin, "The number of burn in samples to throw "\
            "away ({0}) cannot exceed the number of steps that were saved "\
            "({1}). Try setting the burnin keyword argument.".format(burnin,
                                                                     nsteps)
        samples = self.sampler.chain[:, burnin:, :]
        nwalkers, nsteps, ndim = np.shape(samples)
        samps = np.reshape(samples, (nwalkers*nsteps, ndim))
        msamps = mist.interp_value([samps[:, 0], samps[:, 1], samps[:, 2]],
                                   ["mass"])
        m = np.median(msamps)
        errp = np.percentile(msamps, 84) - m
        errm = m - np.percentile(msamps, 16)
        return m, errm, errp, msamps


    def feh_results(self, burnin=0):
        """The metallicity samples.

        The posterior samples for metallicity, optionally with a specified
        number of burn in steps thrown away.

        Args:
            burnin (Optional[int]): The number of samples to throw away.
                Default is 0.

        Returns:
            The median metallicity, its 16th and 84th percentile lower and
            upper uncertainties and the metallicity samples.

        """

        nwalkers, nsteps, ndim = np.shape(self.sampler.chain)
        assert nsteps > burnin, "The number of burn in samples to throw "\
            "away ({0}) cannot exceed the number of steps that were saved "\
            "({1}). Try setting the burnin keyword argument.".format(burnin,
                                                                     nsteps)
        samples = self.sampler.chain[:, burnin:, 2]
        samps = np.reshape(samples, (nwalkers*(nsteps - burnin)))
        f = np.median(samps)
        errp = np.percentile(samps, 84) - f
        errm = f - np.percentile(samps, 16)
        return f, errm, errp, samps


    def distance_results(self, burnin=0):
        """The ln(distance) samples.

        The posterior samples for distance (in natural log, parsecs),
        optionally with a specified number of burn in steps thrown away.

        Args:
            burnin (Optional[int]): The number of samples to throw away.
                Default is 0.

        Returns:
            The median ln(distance), its 16th and 84th percentile lower and
            upper uncertainties and the ln(distance) samples.

        """

        nwalkers, nsteps, ndim = np.shape(self.sampler.chain)
        assert nsteps > burnin, "The number of burn in samples to throw "\
            "away ({0}) cannot exceed the number of steps that were saved "\
            "({1}). Try setting the burnin keyword argument.".format(burnin,
                                                                     nsteps)
        samples = self.sampler.chain[:, burnin:, 3]
        samps = np.reshape(samples, (nwalkers*(nsteps - burnin)))
        d = np.median(samps)
        errp = np.percentile(samps, 84) - d
        errm = d - np.percentile(samps, 16)
        return d, errm, errp, samps


    def Av_results(self, burnin=0):
        """The Av samples.

        The posterior samples for V-band extinction, optionally with
        a specified number of burn in steps thrown away.

        Args:
            burnin (Optional[int]): The number of samples to throw away.
                Default is 0.

        Returns:
            The median Av, its 16th and 84th percentile lower and upper
            uncertainties and the Av samples.

        """

        nwalkers, nsteps, ndim = np.shape(self.sampler.chain)
        assert nsteps > burnin, "The number of burn in samples to throw "\
            "away ({0}) cannot exceed the number of steps that were saved "\
            "({1}). Try setting the burnin keyword argument.".format(burnin,
                                                                     nsteps)
        samples = self.sampler.chain[:, burnin:, 4]
        samps = np.reshape(samples, (nwalkers*(nsteps - burnin)))
        a_v = np.median(samps)
        errp = np.percentile(samps, 84) - a_v
        errm = a_v - np.percentile(samps, 16)
        return a_v, errm, errp, samps
