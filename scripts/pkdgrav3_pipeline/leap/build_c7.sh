#!/bin/bash -l
# pkdgrav3 for leap (and flock, raft): conda environment solved for glibc 2.17 (CentOS/RHEL 7), skylake-avx512 build.
#   bash build_c7.sh [root]   (default /user1/$USER/sahyadri_c7/pkdgrav3_io; run on leap's login node, ~30 min)
# Creates <root>/env (~3 GB; package cache <root>/conda_pkgs), <root>/pkdgrav3 (copy of SRC, default pride's clone:
# the same commit as swarm), <root>/build, <root>/build_snapcomp. Needs Miniforge in $CONDA_BASE.
# leap's login profile sets ARCH=x86_64, so compile_pegasus.sh is run without a login shell (bash -l would reset ARCH
# and cmake would get -march=x86_64).
set -eu
R=${1:-/user1/$USER/sahyadri_c7/pkdgrav3_io}
SRC=${SRC:-/user1/$USER/sahyadri_pride/pkdgrav3_io/pkdgrav3}
mkdir -p "$R" && cd "$R"
export CONDA_BASE=${CONDA_BASE:-/user1/$USER/miniforge3} CONDA_PKGS_DIRS=$R/conda_pkgs CONDA_OVERRIDE_GLIBC=2.17
source "$CONDA_BASE/etc/profile.d/conda.sh"
[ -d pkdgrav3 ] || rsync -a "$SRC/" pkdgrav3/
(cd pkdgrav3 && git log -1 --format="pkdgrav3 source %h %s")
if [ ! -d env ]; then
  conda create -y -q -p "$R/env" --override-channels -c conda-forge \
    python=3.11 "numpy<2" "cython>=3" dill tomli h5py zstandard \
    c-compiler cxx-compiler "sysroot_linux-64=2.17" make cmake pkg-config git \
    "openmpi=5" "fftw=*=mpi_openmpi*" "hdf5=*=nompi*" gsl libboost-devel libhwloc zstd zlib
fi
conda activate "$R/env"
python -c "import numpy, h5py; print('env ok: numpy', numpy.__version__)"
PKD_IO_ROOT=$R CONDA_BASE=$CONDA_BASE ARCH=skylake-avx512 bash pkdgrav3/tools/pegasus/compile_pegasus.sh
# the newest glibc symbol version the binary and the MPI libraries need (must be <= 2.17)
for f in build/pkdgrav3 env/lib/libmpi.so env/lib/libucc.so env/lib/libfabric.so; do
  echo "$f $(objdump -T "$f" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -V | tail -1)"; done
