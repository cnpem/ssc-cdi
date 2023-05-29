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


[0.5.13] - 2023-05-29
--------------------

Added
~~~~~
- Added option to apply flatfield in CAT ptycho after restoration
 

[0.5.12] - 2023-05-29
--------------------

Added
~~~~~
- Added system call to h5clear hdf5 file prior to restoration call



[0.5.11] - 2023-05-25
--------------------

Fixed
~~~~~
- Fixed bug for reading username from system when sending jobs to cluster



[0.5.10] - 2023-05-16
--------------------

Changed
~~~~~
- Refactored code with new folder structure and modules
- Major changes to functions and code cleanup

Added
~~~~~
- CUDA restoration for single and multiple acquisitions



[0.4.16] - 2023-03-07
--------------------
- Added variable to input that can increase ptycho object size by padding
- Bugfixes



[0.4.15] - 2023-03-06
--------------------

Changed
~~~~~
- Changed number of possible GPUs for CAT interfaces for 5 at Cluster and 6 at Local since restructuring of the machines
