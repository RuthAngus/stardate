# import numpy as np
# import stardate as sd


# def test_fit():
#     iso_params = {"teff": (5777, 10),
#                 "logg": (4.44, .05),
#                 "feh": (0., .001),
#                 "parallax": (1., .01)}
#     prot, prot_err = 26., 1.
#     star = sd.Star(iso_params, prot, prot_err)

#     nsteps = 200
#     star.fit(max_n=nsteps)

#     thin_by, nwalkers, ndim = 1, 24, 5

#     burnin = 1
#     star.fit(max_n=nsteps, burnin=burnin)
#     assert np.shape(star.samples) == ((nsteps//thin_by - burnin)*nwalkers,
#                                       ndim)
#     assert np.shape(star.sampler.chain) == (nwalkers, nsteps/thin_by, ndim)


# if __name__ == "__main__":
#     # test_fit()
