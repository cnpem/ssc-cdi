.. _logV0:

Changelog versions 0.Y.Z
========================

Version 0.14.0 - 2024-10-23
---------------------------
* Fixed

    - Bug fixes for use of different initial guesses for object and probe

* Changed

    - Renamed Python function in ptycho engines for better understanding
    - Variable renaming in python codes and input dictionary
    - Variable renaming in CUDA codes for better understanding
    - Python and CUDA modules renamed to follow a similar standard

* Added

    - New RAAR option in CUDA storing temporary wavefront in CPU when not enough VRAM is available
    - New loglikelihood and MSE error metrics in Python engines
    - Added Fourier Power Bound variable to Ptycho python engines

Version 0.13.5 - 2024-10-01
---------------------------
* Fixed

    - Bug fix of slice_visualizer for jupyter notebooks. Vmax and Vmin inputs should be working as well as the colorbar

Version 0.13.4 - 2024-09-27
---------------------------
* Added

    - Added creation of hdf5 output folder path if needed

Version 0.13.3 - 2024-09-26
---------------------------
* Added

    - Added functions to check data types before ptychography
    - Added function to randomly remove scan points from positions and data arrays if wanted

Version 0.13.2 - 2024-09-26
---------------------------
* Changed

    - Refactor of CUDA codes. Removed unused varialbes, renamed variables to more intuitive names and merged some functionality. 

* Fixed

    - Fixed bugs when calling main ptychography function in Python

* Added

    - Added functions to bin data in the ptychography pipeline.
    - Added function to check input dictionary for missing keys and use default values for them, if needed

Version 0.13.1 - 2024-09-20
---------------------------
* Changed

    - Changed cupy dependency to be cupy==12.0

Version 0.13.0 - 2024-09-19
---------------------------
* Added

    - Initial merge of the 3d planewave CDI code. 


Version 0.12.0 - 2024-09-19
---------------------------
* Changed

    - Removed restoration module and beamline specific codes (pipelines) from sscCdi. These are now part of the ssc-cdi-apps package.

Version 0.11.2 - 2024-08-27
---------------------------
* Fixed

    - Fixed bug for PIE when considering incoherent_modes ``> 1``

Version 0.11.1 - 2024-08-01
---------------------------
* Fixed

    - Fixed bug for corrected positions coming out of AP_CUDA engine
    - Temporary fix for missing source_distance key in input dictionary

Version 0.11.0 - 2024-07-31
---------------------------
* Added

    - Added Maximum likelihood algorithm for Ptychography in python using Cupy. It optimizes either with gradient descent or conjugate gradient.

Version 0.10.2 - 2024-07-18
---------------------------
* Changed

    - Changed slice visualizer function for CAT tomo pipeline

Version 0.10.1 - 2024-06-28
---------------------------
* Changed

    - Removed Sigmask parameter
    - Removed Background paramater

Version 0.10.0 - 2024-06-20
---------------------------
* Added

    - Added option for initial guess using model of a Fresnel Zone Plate
    - A module of user friendly plots has been added to Ptychography
    - Added option to use restored flatfield and mask in CAT ptycho pipeline

* Changed

    - New version of input dictionary. Calls have been organized and simplified
    - Unification of Python and CUDA algorithms

* Documentation

    - Part of the documentation has been updated

Version 0.9.6 - 2024-05-29
--------------------------
* Changed

    - Added optional input to cat_restoration, to choose the scaling parameter for the PIMEGA detector.


Version 0.9.5 - 2024-05-29
--------------------------
* Added

    - Added simple interactive function in misc for selecting equalization mask
    - Added equalization by gradient descent method

* Changed

    - Refactored tomo processing functions, mostly changing the calls from a dic input to a direct input call


Version 0.9.4 - 2024-05-16
--------------------------
* Added

    - Prototype for using probe from previous ptycho run in new one to improve reconstruction

Version 0.9.3 - 2024-05-08
--------------------------
* Changed

    - Changed the routine for reading probe positions in EMA pipeline

Version 0.9.2 - 2024-05-02
--------------------------
* Added

    - Alternative method for equalization using non-continuos mask
    - Calculation of phase derivative via hilbert transform for Backprojection without phase unwrapping


Version 0.9.2 - 2024-05-02
--------------------------
* Added

    - Alternative method for equalization using non-continuos mask
    - Calculation of phase derivative via hilbert transform for Backprojection without phase unwrapping

Version 0.9.1 - 2024-04-08
--------------------------
* Fixed

    - Fixed initial guess for probe modes for RAAR_python. Secondary modes are random arrays between 0 and 1.

Version 0.9.0 - 2024-04-05
--------------------------
* Added

    - New nearfield ptychography pipeline for Mogno
    - C++/CUDA codes from sscPtycho were migrated to sscCdi
    - CUDA implementation of ePIE algorithm (single GPU only)

* Changed

    - Bug fixes for Fresnel Ptychography python codes


Version 0.8.10 - 2024-03-26
---------------------------
* Documentation

    - Releasing new major version due to reestructuring of package from version 0.7.15


Version 0.7.17 - 2024-03-26
---------------------------
* Fixed

    - Fixed missing imports of CNB pipeline after refactoring in version 0.7.15

Version 0.7.16 - 2024-03-26
---------------------------
* Fixed

    - Fixed ePIE and RAAR python wrappers for correct algorithm call with new Fresnel propagator
    - Fixed missing imports after refactoring in version 0.7.15

Version 0.7.15 - 2024-03-26
---------------------------
* Changed

    - Restructured package modules, separating beamline specific code into the beamline modules

* Removed

    - Removed dependencies of ssc packages that are pipeline specific. The only dependency that remains in from sscPtycho, which shall be incorporated into sscCdi into the future. 

* Added

    - Added fresnel cone-beam propagator to Python version of RAAR. Fresnel ptychography working in this cases for simulated samples.


Version 0.7.14 - 2024-03-01
---------------------------
* Removed

    - Removed CI/CD for power architecture

* Added

    - Added fresnel cone-beam propagator to Python version of RAAR. Fresnel ptychography working in this cases for simulated samples.

Version 0.7.13 - 2024-02-21
---------------------------
* Removed

    - Remove dev alignment files

Version 0.7.12 - 2024-02-21
---------------------------
* Changed

    - Added EMA crop routine

Version 0.7.11 - 2024-02-19
---------------------------
* Changed

    - Added upgrades to CAT tomography pipeline.
    - The alignment functions (Cross corerlation and Vertical mass fluctuation) were removed from ssc-cdi and transferred to ssc-raft.

Version 0.7.1 - 2024-02-09
--------------------------
* Fixed

    - Reading probe positions bug fixed

Version 0.7.0 - 2024-02-08
--------------------------
* Added

    - Python implementation of ePie and RAAR algorithms in cupy
    - EMA beamline pipeline implementation

Version 0.6.39 - 2024-01-16
---------------------------
* Changed

    - Rectangular final object

Version 0.6.38 - 2024-01-05
---------------------------
* Fixed

    - Optimized combine and save final file routines

Version 0.6.37 - 2023-12-06
---------------------------
* Changed

    - Option to use initial probes and objects from previous ptychography

Version 0.6.36 - 2023-11-28
---------------------------
* Fixed

    - Add option to do not use gradient when using alignment variance field

Version 0.6.35 - 2023-11-22
---------------------------
* Fixed

    - Fixed initial object does not need to be frame zero and supressed output for corrected positions from ptycho function and save volumes function

Version 0.6.34 - 2023-11-17
---------------------------
* Fixed

    - Fixed incoherent modes bug

* Added

    - Save final positions when using position correction algorithm
    - New function to remove bad frames anywhere in tomography pipeline

Version 0.6.33 - 2023-11-10
---------------------------
* Fixed

    - Fixed wrong file index when running ptycho for selected projections

Version 0.6.32 - 2023-08-31
---------------------------
* Added

    - Added scripts for tomo and tif convertion for running with sbatch

Version 0.6.31 - 2023-08-30
---------------------------
* Changed

    - Required installation packages and update of documentation

Version 0.6.30 - 2023-08-28
---------------------------
* Documentation

    - Updated documentation pages

Version 0.6.29 - 2023-08-25
---------------------------
* Documentation

    - Added missing documentation

Version 0.6.28 - 2023-08-22
---------------------------
* Fixed

    - Fixed bug for missing save folder path when performing restoration via IO mode

Version 0.6.27 - 2023-08-22
---------------------------
* Fixed

    - Fixed bug for correcting file reading when performing restoration via IO mode

Version 0.6.26 - 2023-08-21
---------------------------
* Fixed

    - Fixed bug for correcting DP dimension when performing restoration via IO mode

Version 0.6.25 - 2023-08-08
---------------------------
* Fixed

    - Fixed bug for correctly saving ordered angles file

Version 0.6.24 - 2023-08-08
---------------------------
* Fixed

    - Fixed bug when reading angles indices for the cases where ptychography had to be restarted from an intermediate frames

* Added

    - Added new alignment options (Cross Correlation and Vertical Mass Fluctuation) for tomography pipeline, according to https://doi.org/10.1364/OE.27.036637

Version 0.6.23 - 2023-08-02
---------------------------
* Fixed

    - Fixed bug when reading files for specific projections in restoration and ptycho routines

Version 0.6.22 - 2023-07-24
---------------------------
* Added

    - Commented PtyPy imports for now. Need to update Python version to 3.9 in all cluster machines before making it fully available. 


Version 0.6.21 - 2023-07-19
---------------------------
* Added

    - Changes to tomo_processing for using new version 2.2.0 of sscRaft with FBP and EM without regular angles

* Fixed

    - Fixed angle conversion for degrees to radians for tomography


Version 0.6.20 - 2023-07-11
---------------------------
* Added

    - Included wrapper and script for running reconstruction with Ptypy using Caterete data. Only single 2D reconstruction possible for now. 


Version 0.6.19 - 2023-07-07
---------------------------
* Fixed

    - Fixed count of files when doing ptycho from multiple datafolders for determining sinogram dimension

Version 0.6.18 - 2023-07-05
---------------------------
* Added

    - Added option to skip cropping of the diffraction pattern when restoring DP without CUDA

Version 0.6.17 - 2023-07-03
---------------------------
* Added

    - Added new dynamic plotting function to preview both magnitude and phase

* Changed

    - Saving also angles, positions and errors after each iteration and combining them into single volume at output hdf5 file at the end. 

Version 0.6.16 - 2023-06-29
---------------------------
* Added

    - Added new feature to load already restored .npy flatfield. It also does the forward restoration of the flatfield.

Version 0.6.15 - 2023-06-22
---------------------------
* Fixed

    - Fixed bug when for correctly determining sinogram size when running ptycho reconstructions for all frames, that is, with projections = []

Version 0.6.14 - 2023-06-21
---------------------------
* Added

    - Added binning strategies after restoration for CATERETE


Version 0.6.13 - 2023-06-16
---------------------------

* Fixed

    - Fixed bug for clearing multiple open hdf5 files that were not correctly closed by the Pimega backend via h5clear -s command


Version 0.6.12 - 2023-06-07
---------------------------
* Fixed

    - Fixed bug for correctly counting number of frames when doing ptychography for CAT using multiple data folders

Version 0.6.11 - 2023-06-06
---------------------------
* Changed

    - Restructured functions in files for unified restoration between CNB and CAT 
    - Added option for subtraction mask 

* Fixed

    - Fixed bugs in restoration functions

Version 0.6.10 - 2023-06-05
---------------------------
* Added

    - Merged codes for Ptychography both at CATERETE and CARNAUBA beamlines
    - Changed input options for probe support

Version 0.5.13 - 2023-05-29
---------------------------
* Added

    - Added option to apply flatfield in CAT ptycho after restoration
 
Version 0.5.12 - 2023-05-29
---------------------------
* Added

    - Added system call to h5clear hdf5 file prior to restoration call

Version 0.5.11 - 2023-05-25
---------------------------
* Fixed

- Fixed bug for reading username from system when sending jobs to cluster

Version 0.5.10 - 2023-05-16
---------------------------
* Changed

    - Refactored code with new folder structure and modules
    - Major changes to functions and code cleanup

* Added

    - CUDA restoration for single and multiple acquisitions

Version 0.4.16 - 2023-03-07
---------------------------
- Added variable to input that can increase ptycho object size by padding
- Bugfixes


Version 0.4.15 - 2023-03-06
---------------------------
* Changed

    - Changed number of possible GPUs for CAT interfaces for 5 at Cluster and 6 at Local since restructuring of the machines

