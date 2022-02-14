Public
==============

***************
Getting started
***************

Sirius Scientific Computing Group approach for Coherent Diffraction Imaging reconstructions.

**Author (s)**

	Yuri Tonin (yuri.tonin@lnls.br) 
    Júlia Carvalho (julia.carvalho@lnls.br)

Currently, the ssc-cdi module performs Ptycography reconstructions for the Cateretê and Carnaúba beamlines. The reconsutruction codes were developed by Giovanni Baraldi and should be replaced in the future by newer and properly documented versions. In the future, this module will also contain Plane-wave coherent diffraction imaging codes for Cateretê.


Main references
****************

We leave this section to list the main references used for developing the reconstruction algorithms in the future.

    - Example Ref 1: example of ptycography RAAR reconstruction paper.
    - Example Ref 2: etc

***************
Code structure
***************

The main code and many of the necessary functions are present in the sscptycho.py script. Some other functions are contained in three separate modules inside the sscCdi folder:

    - misc.py
    - restauration.py
    - unwrap.py

The docstring description of the functions are list in the sections ahead. In the future, many of the functions contained in the main script should be transferred to separate module.

JSON input file
****************

A JSON file is used to list all the parameters needed for the mainscript (sscpytcho.py).  We detail this input file in the Examples section.

----

****************
Main script
****************

We detail below the functions contained in the main script. This is the script used to run the ptycographic reconstruction. 

Script tasks
************

    - Describe here main actions of the script?

----

.. automodule:: sscptycho
    :members:

****************
Modules
****************

Below we list the functions of the main modules used in the code.

----

Miscellanoeus module
*********************

.. automodule:: misc
    :members:

Restauration module
********************

.. automodule:: restauration
    :members:

Unwrap module
****************

.. automodule:: unwrap
    :members:
