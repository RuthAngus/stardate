import numpy as np
import matplotlib.pyplot as plt
import stardate
from priors import distance_prior
from stardate.lhf import lnprior


def test_lnprior():
    """
    Test that the prior on distance increases monotonically between 0 and 3000
    """
    print(lnprior([355., 9., 0., 1000., .001]))
    distances = np.linspace(0, 3000, 1000)
    probs = []
    for i in range(len(distances)):
        probs.append(distance_prior(distances[i]))
    diffs = np.diff(probs)
    assert diffs.all() > 0


if __name__ == "__main__":
    test_lnprior()
