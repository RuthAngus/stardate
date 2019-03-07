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
`h5py <https://www.h5py.org/>`_, and
`tqdm <https://tqdm.github.io/>`_ and
`isochrones <https://github.com/timothydmorton/isochrones>`_.

The first four of these can be installed using conda or pip:

.. code-block:: bash

    conda install numpy pandas h5py tqdm

or

.. code-block:: bash

    pip install numpy pandas h5py tqdm

You'll also need to download isochrones and switch to the bolo branch:

.. code-block:: bash

    git clone https://github.com/timothydmorton/isochrones
    cd isochrones
    git checkout bolo
    python setup.py install
    
Note that the bolo branch is currently the development branch for the upcoming release of isochrones v2.0, so stay tuned for updates.
