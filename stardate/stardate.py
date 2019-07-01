import os
import numpy as np
import pandas as pd
import h5py
import tqdm
from .lhf import lnprob, lnlike, nll# , ptform
from isochrones import StarModel, SingleStarModel, get_ichrone
import emcee
import scipy.optimize as spo
# from dynesty import NestedSampler

from isochrones.priors import FlatPrior
from isochrones.mist import MIST_Isochrone
bands = ["B", "V", "J", "H", "K", "BP", "RP", "G"]
# mist = MIST_Isochrone(bands)
mist = get_ichrone("mist", bands=bands)
from isochrones.priors import GaussianPrior


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
        prot (Optional[float]): The rotation period of the star in days.
        prot_err (Optional[float]): The uncertainty on the stellar rotation
            period in days.
        Av (Optional[float]): The v-band extinction (if known).
        Av_err (Optional[float]): The v-band extinction uncertainty
            (if known).
        savedir (Optional[str]): The name of the directory where the samples
            will be saved. Default is the current working directory.
        filename (Optional[str]): The name of the h5 file which the posterior
            samples will be saved in.

    """

    def __init__(self, iso_params, prot=None, prot_err=None, Av=None,
                 Av_err=None, savedir=".", filename="samples"):

        if prot is not None:
            if prot <= 0.:
                print("WARNING: rotation period <= 0, isochrone likelihood" \
                    "function will be used")

        if prot_err is not None:
            if prot_err <= 0.:
                print("WARNING: rotation period uncertainty <= 0, isochrone" \
                    "likelihood function will be used")

        self.iso_params = iso_params
        self.prot = prot
        self.prot_err = prot_err
        self.Av_mean = Av
        self.Av_sigma = Av_err
        self.savedir = savedir
        self.filename = filename

    # def dynesty_fit(self, iso_only=False, gyro_only=False, rossby=True,
    #                 model="praesepe"):
    #     """
    #     Run MCMC on a star using dynasty.

    #     Explore the posterior probability density function of the stellar
    #     parameters using MCMC (via dynasty).

    #     Args:
    #         rossby (Optional[bool]): If True, magnetic braking will cease
    #             after Ro = 2. Default is True.
    #         iso_only (Optional[bool]): If true only the isochronal likelihood
    #             function will be used.
    #         gyro_only (Optional[bool]): If true only the gyrochronal
    #             likelihood function will be used. Beware: this may not do what
    #             you might assume it does... In general this setting is not
    #             currently very useful!
    #         rossby (Optional[bool]): If True, magnetic braking will cease
    #             after Ro = 2. Default is True.
    #         model (Optional[bool]): The gyrochronology model to use. The
    #             default is "praesepe" (the Praesepe-based model). Can also be
    #             "angus15" for the Angus et al. (2015) model.

    #     """

    #     mod = StarModel(mist, **self.iso_params)  # StarModel isochrones obj
    #     # mod.set_prior(age=FlatPrior(8, 10.14))

    #     # lnlike arguments
    #     args = [mod, self.prot, self.prot_err, iso_only, gyro_only, rossby,
    #             model]
    #     self.args = args
    #     ndim = 5
    #     sampler = NestedSampler(lnlike, ptform, ndim, logl_args=args,
    #                             nlive=1500, bound="balls")
    #     sampler.run_nested()

    #     # normalized weights
    #     self.weights = np.exp(sampler.results.logwt
    #                           - sampler.results.logz[-1])

    #     self.samples = sampler.results.samples
    #     df = pd.DataFrame({"samples": [self.samples],
    #                        "weights": [self.weights]})
    #     fname = "{0}/{1}.h5".format(self.savedir, self.filename)
    #     df.to_hdf(fname, key="samples", mode="w")

    def fit(self, inits=[329.58, 9.5596, -.0478, 260, .0045],
            nwalkers=24, max_n=100000, thin_by=100, burnin=0, iso_only=False,
            gyro_only=False, optimize=False, rossby=True, model="praesepe",
            seed=None, save_samples=True):
        """Run MCMC on a star using emcee.

        Explore the posterior probability density function of the stellar
        parameters using MCMC (via emcee).

        Args:
            inits (Optional[array-like]): A list of initial values to use for
                EEP, age (in log10[yrs]), feh, distance (in pc) and Av.
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
                likelihood function will be used. Beware: this may not do what
                you might assume it does... In general this setting is not
                currently very useful!
            optimize (Optional[bool]): If True, initial parameters will be
                found via optimization. Default is False.
            rossby (Optional[bool]): If True, magnetic braking will cease
                after Ro = 2. Default is True.
            model (Optional[bool]): The gyrochronology model to use. The
                default is "praesepe" (the Praesepe-based model). Can also be
                "angus15" for the Angus et al. (2015) model.
            seed (Optional[int]): The random number seed. Set this if you want
                to regenerate exactly the same samples each time.
            save_samples (Optional[bool]): saving samples is the computational
                bottleneck. If you want to save time and don't need to save
                the samples using the HDF5 backend, set this to False.

        """

        self.max_n = max_n
        self.nwalkers = nwalkers
        self.thin_by = thin_by
        self.save_samples = save_samples

        if burnin > max_n/thin_by:
            burnin = int(max_n/thin_by/3)
            print("Automatically setting burn in to {}".format(burnin))

        p_init = [inits[0], inits[1], inits[2], np.log(inits[3]), inits[4]]
        self.p_init = p_init

        if seed is not None:
            np.random.seed(seed)

        # Create the directory if it doesn't already exist.
        if save_samples:
            if not os.path.exists(self.savedir):
                os.makedirs(self.savedir)

        # Set up the backend
        # Don't forget to clear it in case the file already exists
        ndim = 5
        if save_samples:
            fn = "{0}/{1}.h5".format(self.savedir, self.filename)
            backend = emcee.backends.HDFBackend(fn)
            backend.reset(nwalkers, ndim)
            self.backend = backend

        # Set up the StarModel object needed to calculate the likelihood.
        mod = SingleStarModel(mist, **self.iso_params)  # StarModel isochrones obj
        # mod.set_prior(age=FlatPrior(bounds=(8, 10.14)))

        # Set up a Gaussian prior with the extinction, if provided.
        if self.Av_mean is not None and self.Av_sigma is not None:
            mod.set_prior(AV=GaussianPrior(self.Av_mean, self.Av_sigma,
                                           bounds=(0, 1)))

        # lnprob arguments

        args = [mod, self.prot, self.prot_err, iso_only, gyro_only, rossby,
                model]
        self.args = args

        # Optimize. Try a few inits and pick the best.
        if optimize:
            neep, nage = 5, 5
            likes1, likes2 = np.zeros(nage), np.zeros(neep)
            likes = np.zeros((neep, nage))
            inds = np.zeros(nage)
            result_list = np.zeros((neep, nage, 5))
            EEPS, AGES = np.meshgrid(np.linspace(200, 500, neep),
                                    np.linspace(7, 10., nage), indexing="ij")

            for i in range(neep):
                for j in range(nage):
                    inits = [EEPS[i, j], AGES[i, j], 0., np.log(1000.), .01]
                    results = spo.minimize(nll, inits, args=args)
                    likes1[j] = lnprob(results.x, *args)[0]
                    likes[i, j] = likes1[j]
                    result_list[i, j, :] = results.x
                inds[i] = np.argmax(likes1)
                likes2[i] = max(likes1)
            eep_ind = np.argmax(likes2)
            age_ind = int(inds[eep_ind])
            self.p_init = result_list[eep_ind, age_ind, :]

        # Run the MCMC
        sampler = self.run_mcmc()
        self.sampler = sampler

        nwalkers, nsteps, ndim = np.shape(self.sampler.chain)
        self.samples = np.reshape(self.sampler.chain[:, burnin:, :],
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

        ndim = len(self.p_init)  # Should always be 5. Hard code it?
        p0 = [np.random.randn(ndim)*1e-4 + self.p_init for j in
              range(self.nwalkers)]

        # Broader gaussian for EEP initialization
        # p0 = np.empty((self.nwalkers, ndim))
        # p0[:, 0] = np.random.randn(self.nwalkers)*1e-4 + self.p_init[0]
        # p0[:, 1] = np.random.randn(self.nwalkers)*1e-4 + self.p_init[1]
        # p0[:, 2] = np.random.randn(self.nwalkers)*1e-4 + self.p_init[2]
        # p0[:, 3] = np.random.randn(self.nwalkers)*1e-4 + self.p_init[3]
        # p0[:, 4] = np.random.randn(self.nwalkers)*1e-4 + self.p_init[4]
        # p0 = list(p0)

        if self.save_samples:
            sampler = emcee.EnsembleSampler(
                self.nwalkers, ndim, lnprob, args=self.args,
                backend=self.backend)

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

        else:
            sampler = emcee.EnsembleSampler(self.nwalkers, ndim, lnprob,
                                            args=self.args)
            sampler.run_mcmc(p0, max_n)

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
        a = np.median((10**samps)*1e-9)
        errp = np.percentile((10**samps)*1e-9, 84) - a
        errm = a - np.percentile((10**samps)*1e-9, 16)
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


def percentiles_from_samps(samps, lp):
    """
    Calculate percentiles and maximum-likelihood values from 1D sample array.

    Args:
        samps (array): The 1D sample array for a single star.
        lp (array): The 1D log-probability array.
    Returns:
        med, errm, errp, std, max_like: The median value, lower and upper
            uncertainties, standard deviation and maximum likelihood value.

    """
    med = np.median(samps)
    std = np.std(samps)
    upper = np.percentile(samps, 84)
    lower = np.percentile(samps, 16)
    errp = upper - med
    errm = med - lower
    max_like = samps[lp == max(lp)][0]

    return med, errm, errp, std, max_like


def load_samples(fname, burnin=0):
    """
    Read in H5 file containing samples and return the best-fit parameters.

    Args:
        fname (str): The full path to the H5 file.
        burnin (Optional[int]): The number of burn in steps to discard.
            Default is 0.

    Returns:
        flatsamps (array): The flattened 2d sample array, useful for making
            corner plots, with log-probability samples
            appended. The shape is Nwalkers*(Nsteps - burnin) x ndim + 1.
            The log-probability samples are the final dimension:
        augmented (array): The 3d sample array, useful for plotting chains.
            With log-probs appended as the extra dimension.
        prior_samples (array): The 3d array of prior samples.
    """

    reader = emcee.backends.HDFBackend(fname, read_only=True)
    samples = reader.get_chain()
    lnprobs = reader.get_log_prob()
    prior_samples = reader.get_blobs()
    nsteps, nwalkers, ndim = np.shape(samples)
    augmented = np.zeros((nsteps, nwalkers, ndim+1))
    augmented[:, :, :-1] = samples
    augmented[:, :, -1] = lnprobs
    nsteps, nwalkers, ndim = np.shape(augmented)
    flatsamps = np.reshape(
        augmented[burnin:, :, :], (nwalkers*(nsteps - burnin), ndim))
    return flatsamps, augmented, np.reshape(prior_samples, nsteps*nwalkers), \
        np.reshape(lnprobs, nsteps*nwalkers)


def read_samples(samps, burnin=0):
    """
    Extract best-fit parameters from samples.

    Args:
        samples (array): The 2D sample array (steps x dimensions), optionally
            with log-probability samples appended as an extra dimension.
            Designed to take the output from load_samples.
        burnin (Optional[int]): The number of burn in steps to discard.
            Default is 0.

    Returns:
        param_dict (dict): A pandas dataframe of results with columns:
            EEP_med, EEP_errm, EEP_errp, EEP_std, EEP_ml, age_med, age_errm,
            age_errp, age_std, age_ml, feh_med, feh_errm, feh_errp, feh_std,
            feh_ml, distance_med, distance_errm, distance_errp, distance_std,
            distance_ml, av_med, av_errm, av_errp, av_std, av_ml.
            Columns ending in "ml" mean the maximum-likelihood value. Columns
            ending in "med" mean the median value. Recommend using the ml
            value for stars hotter (lower) than 1.3 in Gaia G_BP - G_RP and
            the median value for stars cooler (higher) than 1.3.
            Age is provided in units of Gyrs and distance in parsecs.
    """

    e, em, ep, _estd, eml = percentiles_from_samps(
        samps[burnin:, 0], samps[burnin:, 5])
    a, am, ap, _astd, aml = percentiles_from_samps(
        (10**samps[burnin:, 1])*1e-9, samps[burnin:, 5])
    f, fm, fp, _fstd, fml = percentiles_from_samps(
        samps[burnin:, 2], samps[burnin:, 5])
    d, dm, dp, _dstd, dml = percentiles_from_samps(
        np.exp(samps[burnin:, 3]), samps[burnin:, 5])
    av, avm, avp, _avstd, avml = percentiles_from_samps(
        samps[burnin:, 4], samps[burnin:, 5])

    param_dict = pd.DataFrame(dict({
        "EEP_med": e, "EEP_errm": em, "EEP_errp": ep, "EEP_std": _estd,
        "EEP_ml": eml,
        "age_med_gyr": a, "age_errm": am, "age_errp": ap, "age_std": _astd,
        "age_ml_gyr": aml,
        "feh_med": f, "feh_errm": fm, "feh_errp": fp, "feh_std": _fstd,
        "feh_ml": fml,
        "distance_med_pc": d, "distance_errm": dm, "distance_errp": dp,
        "distance_std_pc": _dstd, "distance_ml": dml,
        "Av_med": av, "Av_errm": avm, "Av_errp": avp, "Av_std": _avstd,
        "Av_ml": avml}, index=[0]))

    return param_dict
