Installation
=============

This package requires Python3 and CUDA.

Besides standard Python packages, the following need to be installed:

    - SharedArray
    - tqdm
    - scikit-image

Furthermore, sscCdi makes us of other **ssc** packages. Make sure to have all of them installed as well.

    - sscPtycho v2.1.4


Install via pip
**********************

The main one is via our pip server.  For that, you will need to be connected to the CNPEM network.

If this is the first time you are using the Scientific Computing Group pip server, you will need to configure using the following commands:

.. code-block:: bash

    pip config --user set global.extra-index-url http://gcc.lnls.br:3128/simple/
    pip config --user set global.trusted-host gcc.lnls.br


Then, you can simply run

.. code-block:: bash

    pip install sscCdi

If you want to install a specific version, say 0.6.0 do it like

.. code-block:: bash

    pip install sscCdi==0.6.0

Cloning the repository
**********************

The second option is to directly clone sscCdi git repository from `gitlab <https://gitlab.cnpem.br/GCC/ssc-cdi.git>`_ (if you have access to it). 

To install, go to the ssc-cdi folder and install using setup.py.


.. code-block:: bash

    cd ssc-cdi
    python3 setup.py install --cuda --user
