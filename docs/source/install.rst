Installation
=============

Requirements
**********************

This package requires: 

Python3 and CUDA.

Besides standard Python packages, the following need to be installed:

    - tqdm
    - scikit-image


Installation using LNLS network
**********************

If you are connected to LNLS/CNPEM network or using one of our cluster machines, you can simply install using our internal pip server:

.. code-block:: bash

    pip install sscCdi

If you want to install a specific version, say 0.6.0 do it like

.. code-block:: bash

    pip install sscCdi==0.6.0

If you intend to make use of an older version, make sure to uninstall any recent version by running:

.. code-block:: bash

    pip uninstall sscCdi -y


Instalattion from source code 
**********************

To install sscCdi using the source code, enter ssc-cdi folder and install via pip install:

.. code-block:: bash

    cd ssc-cdi
    python3 -m pip install . --user

