#!/usr/bin/env bash
# 3GRadar Python dependencies.   Run:  bash install_deps.sh
#
# IMPORTANT: openEMS and CSXCAD are NOT on PyPI -- they are the FDTD engine's Python
# bindings and must be built from source (no ARM64 wheels either). See fpc3_EC2.md
# for the build outline. This script installs only the pip-available packages.

set -e

# Core: required for fpc3_build / the DE optimizer / validation / diagnostics.
pip install numpy scipy matplotlib

# Optional: MoM/BEM cross-check scripts (fpc_bempp_*.py). Heavy (pulls pyopencl + numba,
# needs an OpenCL runtime) and NOT needed to run the openEMS optimizer -- uncomment if used.
# pip install bempp-cl

echo "Done. Now verify openEMS is importable:  python3 -c 'import openEMS, CSXCAD; print(\"ok\")'"
