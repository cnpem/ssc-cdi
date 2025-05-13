Install
=======

This package uses ``C``, ``C++``, ``CUDA`` and ``Python3``. 
Before installation, you will need the following packages installed:

* ``CUDA >= 10.0.0``
* ``C``
* ``C++`` 
* ``Python >= 3.8.0``
* ``PIP``
* ``libcurl4-openssl-dev``
* ``scikit-build>=0.17.0``
* ``setuptools>=60.0.0``
* ``ninja==1.11.1.1``
* ``wheel==0.45.0``
* ``CMAKE>=3.18``

This package supports nvidia ``GPUs`` with capabilities ``7.0`` or superior and a compiler with support to ``c++17``.

See bellow for build requirements and dependencies.

The library sscCdi can be installed from the source code at `zenodo website <https://zenodo.org/>`_, by our public github or by ``pip``/ ``gitlab``
if inside the CNPEM network. More information on the ``sscCdi`` package on 
`sscCdi website <https://gcc.lnls.br/wiki/docs/ssc-cdi/>`_
available inside the CNPEM network.

Documentation
*************

The ``sscCdi`` package information can be found on the `sscCdi website <https://gcc.lnls.br/wiki/docs/ssc-cdi/>`_ inside the CNEPM network.
Also, the `HTML` documentation can be found in the source directory `./docs/build/index.html` and can be opened in your preferred brownser.


GITHUB
******

The latest package version can be cloned from our public `github <https://github.com/cnpem/ssc-cdi/>`_ repository and installed locally with:

.. code-block:: bash

    git clone https://github.com/cnpem/ssc-cdi.git
    cd ssc-cdi 
    make clean && make

To install a specific version (``<version>``), one can use:

.. code-block:: bash

    git clone  https://github.com/cnpem/ssc-cdi.git --branch v<version> --single-branch
    cd ssc-cdi 
    make clean && make


Example, to install version 0.14.2:

.. code-block:: bash

    git clone https://github.com/cnpem/ssc-cdi.git --branch v0.14.2 --single-branch
    cd ssc-cdi 
    make clean && make


Source code from Zenodo
***********************

The source code can be downloaded from `zenodo website <https://zenodo.org/>`_ under the 
DOI: `10.5281/zenodo.13693177 <https://doi.org/10.5281/zenodo.13693177>`_. On the left panel, one can find
the available versions. Select the version want and download the ``ssc-cdi.tar.gz`` with the source files, 
one can decompress by

.. code-block:: bash

    tar -xvf ssc-cdi.tar.gz


To compile the source files, enter the following command inside the folder

.. code-block:: bash

    make clean && make


PIP
***

One can install the latest version of sscCdi directly from our ``pip server`` inside the CNPEM network.

.. warning::

    This installation option is available only inside the CNPEM network.

.. code-block:: bash

    pip install sscCdi==version --index-url https://gitlab.cnpem.br/api/v4/projects/1875/packages/pypi/simple


Where ``version`` is the version number of the ``sscCdi``. Example:

.. code-block:: bash

    pip install sscCdi==0.14.2 --index-url https://gitlab.cnpem.br/api/v4/projects/1875/packages/pypi/simple


GITLAB
******

.. warning::

    For this installation option is available only inside the CNPEM network.

The latest package version can be cloned from CNPEM's `gitlab <https://gitlab.cnpem.br/>`_ and installed locally with:

.. code-block:: bash

    git clone https://gitlab.cnpem.br/GCC/ssc-cdi.git
    cd ssc-cdi 
    make clean && make

To install a specific version (``<version>``), one can use:

.. code-block:: bash

    git clone  https://gitlab.cnpem.br/GCC/ssc-cdi.git --branch v<version> --single-branch
    cd ssc-cdi 
    make clean && make


Example, to install version 0.14.2:

.. code-block:: bash

    git clone https://gitlab.cnpem.br/GCC/ssc-cdi.git --branch v0.14.2 --single-branch
    cd ssc-cdi 
    make clean && make


Memory
******

Be careful using GPU functions due to memory allocation.

Requirements
************

Before installation, you will need to have the following packages installed:

* ``CUDA >= 10.0.0``
* ``C``
* ``C++`` 
* ``Python >= 3.8.0``
* ``PIP``
* ``libcurl4-openssl-dev``

The build requirements are:

* ``CUBLAS``
* ``CUFFT``
* ``PTHREADS``
* ``scikit-build>=0.17.0``
* ``setuptools>=60.0.0``
* ``ninja==1.11.1.1``
* ``wheel==0.45.0``
* ``CMAKE>=3.18``

The ``Python3`` dependencies are:

* ``numpy<2.0.0>``
* ``scikit-image``
* ``scipy``
* ``matplotlib``
* ``SharedArray``
* ``cupy``
* ``h5py``
* ``ipywidgets``
* ``tqdm``

Uninstall
*********

To uninstall ``sscCdi`` use the command

.. code-block:: bash

    pip uninstall sscCdi
    