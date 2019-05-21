Installation
============

Currently the best way to install *stardate* is from github.

From source:

.. code-block:: bash

    git clone https://github.com/RuthAngus/stardate.git
    cd stardate
    python setup.py install

Dependencies
------------

The dependencies of *stardate* are
`NumPy <http://www.numpy.org/>`_,
`pandas <https://pandas.pydata.org/>`_,
`h5py <https://www.h5py.org/>`_,
`numba <http://numba.pydata.org/>`_,
`tqdm <https://tqdm.github.io/>`_ and
`isochrones <https://github.com/timothydmorton/isochrones>`_.

The first five of these can be installed using conda or pip:

.. code-block:: bash

    conda install numpy pandas h5py numba tqdm

or

.. code-block:: bash

    pip install numpy pandas h5py numba tqdm

You'll also need to download isochrones:

.. code-block:: bash

    git clone https://github.com/timothydmorton/isochrones
    cd isochrones
    python setup.py install
