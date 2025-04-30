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