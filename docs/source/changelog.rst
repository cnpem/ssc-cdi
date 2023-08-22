Changelog
=========
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Types of changes:
 - *Added* for new features.
 - *Changed* for changes in existing functionality.
 - *Deprecated* for soon-to-be removed features.
 - *Removed* for now removed features.
 - *Fixed* for any bug fixes.
 - *Security* in case of vulnerabilities.

[0.6.28] - 2023-08-22
--------------------

Fixed
~~~~~
- Fixed bug for missing save folder path when performing restoration via IO mode

[0.6.27] - 2023-08-22
--------------------

Fixed
~~~~~
- Fixed bug for correcting file reading when performing restoration via IO mode

[0.6.26] - 2023-08-21
--------------------

Fixed
~~~~~
- Fixed bug for correcting DP dimension when performing restoration via IO mode

[0.6.25] - 2023-08-08
--------------------

Fixed
~~~~~
- Fixed bug for correctly saving ordered angles file

[0.6.24] - 2023-08-08
--------------------

Fixed
~~~~~
- Fixed bug when reading angles indices for the cases where ptychography had to be restarted from an intermediate frames

Added
~~~~~
- Added new alignment options (Cross Correlation and Vertical Mass Fluctuation) for tomography pipeline, according to https://doi.org/10.1364/OE.27.036637

[0.6.23] - 2023-08-02
--------------------

Fixed
~~~~~
- Fixed bug when reading files for specific projections in restoration and ptycho routines

[0.6.22] - 2023-07-24
--------------------

Added
~~~~~
- Commented PtyPy imports for now. Need to update Python version to 3.9 in all cluster machines before making it fully available. 


[0.6.21] - 2023-07-19
--------------------

Added
~~~~~
- Changes to tomo_processing for using new version 2.2.0 of sscRaft with FBP and EM without regular angles


Fixed
~~~~~
- Fixed angle conversion for degrees to radians for tomography


[0.6.20] - 2023-07-11
--------------------

Added
~~~~~
- Included wrapper and script for running reconstruction with Ptypy using Caterete data. Only single 2D reconstruction possible for now. 


[0.6.19] - 2023-07-07
--------------------

Fixed
~~~~~
- Fixed count of files when doing ptycho from multiple datafolders for determining sinogram dimension

[0.6.18] - 2023-07-05
----------------------------

Added
~~~~~
- Added option to skip cropping of the diffraction pattern when restoring DP without CUDA

[0.6.17] - 2023-07-03
----------------------------

Added
~~~~~~~~~~
- Added new dynamic plotting function to preview both magnitude and phase

Changed
~~~~~~~~~~
- Saving also angles, positions and errors after each iteration and combining them into single volume at output hdf5 file at the end. 

[0.6.16] - 2023-06-29
----------------------------

Added
~~~~~~~~~~
- Added new feature to load already restored .npy flatfield. It also does the forward restoration of the flatfield.

[0.6.15] - 2023-06-22
----------------------------

Fixed
~~~~~~~~~~
- Fixed bug when for correctly determining sinogram size when running ptycho reconstructions for all frames, that is, with projections = []

[0.6.14] - 2023-06-21
----------------------------

Added
~~~~~~~~~~
- Added binning strategies after restoration for CATERETE


[0.6.13] - 2023-06-16
----------------------------

Fixed
~~~~~~~~~~
- Fixed bug for clearing multiple open hdf5 files that were not correctly closed by the Pimega backend via h5clear -s command


[0.6.12] - 2023-06-07
----------------------------

Fixed
~~~~~~~~~~
- Fixed bug for correctly counting number of frames when doing ptychography for CAT using multiple data folders

[0.6.11] - 2023-06-06
----------------------------

Changed
~~~~~~~~~~
- Restructured functions in files for unified restoration between CNB and CAT 
- Added option for subtraction mask 

Fixed
~~~~~~~~~~
- Fixed bugs in restoration functions


[0.6.10] - 2023-06-05
----------------------------

Added
~~~~~~~~~~
- Merged codes for Ptychography both at CATERETE and CARNAUBA beamlines
- Changed input options for probe support

[0.5.13] - 2023-05-29
----------------------------

Added
~~~~~~~~~~
- Added option to apply flatfield in CAT ptycho after restoration
 

[0.5.12] - 2023-05-29
----------------------------

Added
~~~~~~~~~~
- Added system call to h5clear hdf5 file prior to restoration call



[0.5.11] - 2023-05-25
----------------------------

Fixed
~~~~~~~~~~
- Fixed bug for reading username from system when sending jobs to cluster



[0.5.10] - 2023-05-16
----------------------------

Changed
~~~~~~~~~~
- Refactored code with new folder structure and modules
- Major changes to functions and code cleanup

Added
~~~~~~~~~~
- CUDA restoration for single and multiple acquisitions



[0.4.16] - 2023-03-07
----------------------------
- Added variable to input that can increase ptycho object size by padding
- Bugfixes



[0.4.15] - 2023-03-06
----------------------------

Changed
~~~~~~~~~~
- Changed number of possible GPUs for CAT interfaces for 5 at Cluster and 6 at Local since restructuring of the machines
