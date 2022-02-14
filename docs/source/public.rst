Public
==============

***************
Code structure
***************

The main code and many of the necessary functions are part of the  **sscptycho.py** script. Some other functions are contained in three separate modules inside the sscCdi folder:

    - misc.py
    - restauration.py
    - unwrap.py

The commented functions are listed in the sections ahead. In the future, many of the functions contained in the main script should be transferred to separate modules.

A JSON file is used to insert all the parameters needed for the main script (sscpytcho.py).  We detail this input file in the Examples section.

----

****************
Main script
****************

We detail below the functions contained in the main script. This is the script used to run the ptycographic reconstruction. 

.. automodule:: sscptycho
    :members:

****************
Modules
****************

The three modules the code currently uses are:
    - misc.py: general functions for plotting, dealing with lists, etc
    - restauration.py: functions for calling sscPimega restauration module
    - unwrap.py: functions related to phase unwrapping of 2d images

Miscellaneous module
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
