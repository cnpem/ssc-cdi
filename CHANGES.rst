Version 0.15.0 - 2025-05-13
---------------------------
*Added:*
    - New examples for Ptycho2D and Ptycho-Tomo as jupyter notebooks on ``examples/`` folder.
    - Mean squared error metric computation in CUDA engines.
    - Zenodo ``json`` file.

*Fixed:*
    - Position correction function on ``CUDA`` engines.
    - CI/CD ``build_docs``.
    - Minor issues.

*Changed*
    - ``examples/`` folder.
    - ``README.md``.
    - Merge ``cditypes_planewave.py`` to ``cditypes.py``.

*Removed*
    - ``bcdi/`` dummy folder.
    - ``data/`` and ``examples/`` folder related to Planewave. Moved to ``/ibira/lnls/labs/gcc/data/pwcdi/ssc-cdi-0.14.2-2025-05-07/``.
    - ``data/`` and ``examples/`` folder related to Ptycho. Moved to ``/ibira/lnls/labs/gcc/data/ptycho/Jupyter-notebooks-ssc-cdi-0.14.2-2025-05-07/``.
    - ``sscCdi/ptycho/dev/`` folder. Moved to ``/ibira/lnls/labs/gcc/data/ptycho/Jupyter-notebooks-ssc-cdi-0.14.2-2025-05-07/``.
    - ``sscCdi/planewave/dev/`` folder. Moved to ``/ibira/lnls/labs/gcc/data/pwcdi/ssc-cdi-0.14.2-2025-05-07/``.
    - ``sscCdi/processing/dev/`` folder. Moved to ``/ibira/lnls/labs/gcc/data/ptycho/Jupyter-notebooks-ssc-cdi-0.14.2-2025-05-07/``.

*Documentation*
    - Upgrade on pages, install, changelog and home.
    - Added Ptycho engines call function API.

Version 0.14.2 - 2024-12-17
---------------------------
*Added:*
    - Poisson Negative Log-Likelihood error metric computation in CUDA engines
    - Mean squared error metric computation in CUDA engines

Version 0.14.1 - 2024-11-26
---------------------------
*Fixed:*
    - Bug fix: The array of corrected positions is now copied to the right variable
    - Bug fix: Removed cupy alias that was breaking some numpy calls
    - Bug fix: Ninja was taking forever to run. Pinned the ninja version to 1.11.1.1.
