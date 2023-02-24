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


[0.3.5] - 2022-12-15
--------------------
Fixed
~~~~~
- Fixed selection of diffraction pattern center in Jupyter interface. Coordinates are correct in the GUI and inside the code for restoration

Changed
~~~~~
- Removed load input buttons from Ptycho and Tomo GUIs. Last used input by the user will now be loaded at startup of the GUI

[0.3.0] - 2022-10-29
--------------------

Added
~~~~~
- Added button to load the last saved input file, both for tomo and ptycho interfaces
- Added dictionary input containing packages' versions used when running algorithms

Fixed
~~~~~
- Fixed bug of tomography requiring to run wiggle beforehand. A file containing wiggle center of mass is now saved to be read later on.

Changed
~~~~~
- Changed input folders and logfiles to contain the username
- Changed widgets disposition for a better user experience in both tomo and ptycho interfaces
- Changed quantity of plots shown in tomo interface tabs. The user can now select which direction of the slices to visualize, so that less plots are loaded.
