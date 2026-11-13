#!/bin/bash -l
# One-time setup of the halo codes and Python environment of the Sahyadri pipeline on pegasus (IUCAA): the same
# builds as setup_pawna.sh (Rockstar with the .pkc reader, consistent-trees, venv, post-process C helpers), with
# pkdgrav3's conda environment for compilers and HDF5 (pegasus has no system HDF5).
#
# usage: setup_pegasus.sh <CODES dir> <PKD_IO_ROOT>
#   PKD_IO_ROOT  pkdgrav3 set up with pkdgrav3's tools/pegasus/setup_pegasus.sh (conda env in $PKD_IO_ROOT/env,
#                snapcomp build with zstd); e.g. ~/sahyadri_pipeline/pkdgrav3_io
#   CODES        e.g. ~/sahyadri_pipeline/codes (the home quota is 150 GB: pip's cache goes to PIP_CACHE_DIR)
set -eu
CODES=${1:?usage: setup_pegasus.sh <CODES dir> <PKD_IO_ROOT>}
PKD_IO_ROOT=${2:?usage: setup_pegasus.sh <CODES dir> <PKD_IO_ROOT>}
set +u; source "$PKD_IO_ROOT/pkdgrav3/tools/pegasus/env_pegasus.sh"; set -u
unset PYTHONHOME                    # set by env_pegasus.sh for pkdgrav3's embedded Python; would break the venv
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/data/project/hpc2502016/shadab/cache/pip}
# HDF5 from the conda env, found at run time through the rpath (Rockstar jobs do not load the env)
export HDF5_INC=$CONDA_PREFIX/include HDF5_LIB="$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
export ZSTD_INC=$CONDA_PREFIX/include ZSTD_LIB="$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"   # shared libzstd only
exec bash "$(dirname "$(readlink -f "$0")")/setup_pawna.sh" "$CODES" "$PKD_IO_ROOT"
