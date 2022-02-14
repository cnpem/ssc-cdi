Install
=======

The prerequisite for installing sscFizeau is ``Python 3``.  It can be done using the ssc-cdi ``git`` repository. 

The following Python packages need to be installed:
    - scipy
    - numpy
    - matplotlib
    - os 
    - time
    - h5py
    - sys 
    - pandas 
    - math 
    - PIL 

together with the following ssc modules:

    - sscResolution
    - sscPtycho
    - sscCdi
    - sscIO
    - sscPimega

***
GIT
***

One may clone our `gitlab <https://gitlab.cnpem.br/GCC/ssc-cdi.git>`_ repository and install it using 

.. code-block:: bash

    python3 setup.py install --cuda --user
