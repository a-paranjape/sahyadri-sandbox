#!/bin/bash
# GPU (CUDA) build of pkdgrav3 for the pegasus GPU nodes (gpu001-004: one Tesla P100 16 GB each, compute capability 6.0,
# driver 525.60.13 = CUDA 12.0). A separate build next to the CPU build: $PKD_IO_ROOT/build_gpu/pkdgrav3 (the CPU build
# $PKD_IO_ROOT/build is not touched). Simulations use it with SIM_GPU=1 in run.conf (setup_production.py --sim-gpu;
# planned_runs.txt option gpu=1).
#
#   bash build_pkdgrav3_gpu_pegasus.sh            # on the pegasus login node (needs the internet for conda)
#   PKD_IO_ROOT=... NJ=8 bash build_pkdgrav3_gpu_pegasus.sh
#
# Compilers: CUDA 12.0's nvcc accepts host gcc <= 12, and gcc 12's headers (amxtileintrin.h, avx512fp16intrin.h) do
# not compile under nvcc 12.0, so gcc 11. The WHOLE build must use the same gcc 11: a build with the CPU environment's
# gcc 15 for C++ and gcc 11 only for the CUDA host code crashes at the first GPU work package
# (mdl::CUDA::launch / mdl::gpu::Client::flushCompleted, SIGSEGV). MPI, HDF5, FFTW, GSL and Python come from the CPU
# build's conda environment ($PKD_IO_ROOT/env, tools/pegasus/env_pegasus.sh), so both builds run with the same
# run-time environment. CUDA toolkit and gcc 11: conda environment $PKD_IO_ROOT/env_cuda (created here).
#
# Measured (2026-11-16, 512^3 L200 restart at z = 0.06): one GPU node 10.3 s/step against 36.1 s on a CPU node (2 MPI
# ranks x 16 threads both), 100 % of the gravity tiles on the GPU; 2 GPU nodes 5.7 s/step; snapshots agree with the
# CPU build to better than two CPU runs with different thread counts (positions rms 0.010 kpc/h, P(k) < 2e-7).
# amgr charges a GPU like one more core (a GPU node-hour = 33 units).
set -eu
PKD_IO_ROOT=${PKD_IO_ROOT:-/mnt/home/project/calam.shadab/sahyadri_pipeline/pkdgrav3_io}
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}
NJ=${NJ:-8}
ARCH=${ARCH:-skylake-avx512}          # the GPU nodes' CPUs (Xeon Gold 6142)
CUDA_ARCH=${CUDA_ARCH:-60}            # P100
CUDA_LABEL=${CUDA_LABEL:-cuda-12.0.1} # newest toolkit the 525 driver runs without the compat package
C=$PKD_IO_ROOT/env_cuda
B=$PKD_IO_ROOT/build_gpu
export CONDA_PKGS_DIRS=$PKD_IO_ROOT/conda_pkgs
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ ! -x "$C/bin/nvcc" ] || ! "$C/bin/x86_64-conda-linux-gnu-g++" --version 2>/dev/null | head -1 | grep -q ' 11\.'; then
  echo "creating $C (CUDA 12.0 nvcc + gcc 11)"
  conda create -y -q -p "$C" --override-channels -c "nvidia/label/$CUDA_LABEL" -c conda-forge \
    cuda-nvcc=12.0 cuda-cudart-dev=12.0 cuda-cudart-static=12.0 gcc_linux-64=11 gxx_linux-64=11 > "$PKD_IO_ROOT/env_cuda.log" 2>&1
fi
"$C/bin/nvcc" --version | tail -2
"$C/bin/x86_64-conda-linux-gnu-g++" --version | head -1

# run-time environment of the CPU build (conda MPI, HDF5, ...); sets CONDA_PREFIX=$PKD_IO_ROOT/env
export PKD_IO_ROOT
source "$PKD_IO_ROOT/pkdgrav3/tools/pegasus/env_pegasus.sh"
cd "$PKD_IO_ROOT/pkdgrav3"
git submodule update --init
echo "pkdgrav3 source $(git log -1 --format='%h %s')"
rm -rf "$B"
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DTARGET_ARCHITECTURE="$ARCH" \
  -DCMAKE_C_COMPILER="$C/bin/x86_64-conda-linux-gnu-gcc" -DCMAKE_CXX_COMPILER="$C/bin/x86_64-conda-linux-gnu-g++" \
  -DCMAKE_CUDA_COMPILER="$C/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$C/bin/x86_64-conda-linux-gnu-g++" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCUDAToolkit_ROOT="$C" \
  -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" > "$PKD_IO_ROOT/build_gpu.cmake.log" 2>&1
grep -q '^#define USE_CUDA 1' "$B/pkd_config.h" || { echo "CUDA not enabled (see $PKD_IO_ROOT/build_gpu.cmake.log)"; exit 1; }
cmake --build "$B" -j "$NJ" > "$PKD_IO_ROOT/build_gpu.log" 2>&1
ldd "$B/pkdgrav3" | grep -q libcudart || { echo "$B/pkdgrav3 is not linked with libcudart"; exit 1; }
ls -la "$B/pkdgrav3"
echo "GPU build done: $B/pkdgrav3 (run on a GPU node: -l select=...:ngpus=1; the log reports 'on the GPU: ... ratio')"
