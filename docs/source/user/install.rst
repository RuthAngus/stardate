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
`emcee3 <https://emcee.readthedocs.io/en/latest/`_,
`tqdm <https://tqdm.github.io/>`_ and
`isochrones <https://github.com/timothydmorton/isochrones>`_.

These can be installed using pip:

.. code-block:: bash

    pip install numpy pandas h5py numba "emcee==3.0rc2" tqdm isochrones

.. You'll also need to download isochrones:

.. .. code-block:: bash

..     git clone https://github.com/timothydmorton/isochrones
..     cd isochrones
..     python setup.py install

You can check out the
`isochrones <https://isochrones.readthedocs.io/en/latest/index.html>`_
documentation if you run into difficulties installing that.
