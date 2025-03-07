#!/bin/bash

# Set thread limits for various parallel computing libraries (run before running scattering_ensemble.py if it is too slow)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Thread limits set successfully."