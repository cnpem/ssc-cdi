Installation
=============

Requirements
********************************************


This package requires a machine with NVIDIA CUDA installed. 

To make use of all its functionalities, you will also need to install two other packages from the Sirius Scientific Group: 
    - sscRaft 
    - sscPimega

Installation using LNLS network
********************************************


If you are connected to LNLS/CNPEM network or using one of its cluster machines, you can simply install using our internal pip server:

.. code-block:: bash

    pip install sscCdi

If you want to install a specific version, say 0.10.0 do it like

.. code-block:: bash

    pip install sscCdi==0.10.0

If you intend to make use of an older version, make sure to uninstall any recent version by running:

.. code-block:: bash

    pip uninstall sscCdi -y


Instalattion from source code 
********************************************


To install sscCdi using the source code:

.. code-block:: bash

    cd ssc-cdi
    python3 -m pip install . --user

