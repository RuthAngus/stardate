Installation
============

You can either install *stardate* from source or using pip.

::
    >> git clone https://github.com/RuthAngus/stardate.git
    >> cd stardate
    >> python setup.py install

or
:: pip install stardate_code

Dependencies
------------

The dependencies of *stardate* are
`NumPy <http://www.numpy.org/>`_,
`pandas <https://pandas.pydata.org/>`_,
`h5py <https://www.h5py.org/>`_, and
`tqdm <https://tqdm.github.io/>`_.

These can be installed using conda or pip:

.. code-block:: bash

    conda install numpy pandas h5py tqdm

or

.. code-block:: bash

    pip install numpy pandas h5py tqdm

You'll also need to download isochrones and switch to the eep branch:

.. code-block:: bash
    >> git clone https://github.com/timothydmorton/isochrones
    >> cd isochrones
    >> git checkout eep
    >> python setup.py install
