Installation
=============

The prerequisite for installing ssc-cdi is ``Python 3``.  It can be done using the ssc-cdi ``git`` repository. 

The following Python packages need to be installed:
    - numpy
    - scipy
    - matplotlib
    - h5py
    - scikit-image

Furthermore, the following **ssc** packages must also be installed.

    - sscPtycho
    - sscPimega
    - sscResolution
    - sscIO
    - sscRaft


There are two main approaches for installing **ssc-cdi**. 


***
Install via pip
***

The main one consists in using pip. 

.. code-block:: bash

    pip config --user set global.extra-index-url http://gcc.lnls.br:3128/simple/
    pip config --user set global.trusted-host gcc.lnls.br

***
Cloning the repository
***

One may clone our `gitlab <https://gitlab.cnpem.br/GCC/ssc-cdi.git>`_ repository and install it using:

    python3 setup.py install --cuda --user
