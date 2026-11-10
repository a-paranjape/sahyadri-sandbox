#!/bin/bash
# One-time setup of the halo codes and Python environment of the Sahyadri pkdgrav3 pipeline on pawna.
#
# usage: setup_pawna.sh <CODES dir> [PKD_IO_ROOT]
#   CODES        where to build (e.g. /storage/$USER/sahyadri_codes); the run configs point to it
#   PKD_IO_ROOT  pkdgrav3 install (default /storage/$USER/pkdgrav3_io; set it up first with pkdgrav3's
#                tools/pawna/setup_pawna.sh, see pkdgrav3 docs/running_on_pawna.md). Its snapcomp build
#                provides the zstd library that Rockstar's .pkc reader links against.
# Builds (each from the upstream source plus the changed files kept in this repository under code/):
#   $CODES/rockstar   Rockstar 36ce9eea36ee + Sahyadri changes (PKC reader, curvature, w0-wa), make with_hdf5_pkc
#   $CODES/ctrees     consistent-trees 7c20add + Sahyadri changes (w0-wa/curvature background), make
#   $CODES/venv       Python 3 venv (system site packages) + numpy scipy pandas h5py fitsio zstandard
#   $CODES/lib        cic_update.so, ngp_update.so for post-process (found through SAHYADRI_LIBDIR)
# Rerunning rebuilds everything (the clones are kept).
set -eu
CODES=${1:?usage: setup_pawna.sh <CODES dir> [PKD_IO_ROOT]}
PKD_IO_ROOT=${2:-/storage/$USER/pkdgrav3_io}
SB=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)     # this repository
mkdir -p "$CODES/lib"; CODES=$(cd "$CODES" && pwd)
ZSRC=$PKD_IO_ROOT/build_snapcomp/_deps/snapcomp_zstd_src-src/lib
ZBLD=$PKD_IO_ROOT/build_snapcomp/_deps/snapcomp_zstd_src-build/lib
[ -f "$ZSRC/zstd.h" ] && [ -f "$ZBLD/libzstd.a" ] || { echo "zstd not found under $PKD_IO_ROOT/build_snapcomp (build pkdgrav3 first)"; exit 1; }
overlay() {  # copy the repository's changed files (not the .bak originals) over an upstream checkout
  (cd "$1" && find . -type f ! -name '*.bak' -print0 | xargs -0 -I{} cp --parents {} "$2")
}

echo "== Rockstar"
[ -d "$CODES/rockstar/.git" ] || git clone -q https://bitbucket.org/gfcstanford/rockstar.git "$CODES/rockstar"
(cd "$CODES/rockstar" && git checkout -q -f 36ce9eea36eeda4c333acf56f8bb0d40ff0df2a1 && git clean -qfdx)
overlay "$SB/code/Rockstar/gfcstanford-rockstar-36ce9eea36ee" "$CODES/rockstar"
# HDF5 (Gadget-4/AREPO input, not needed for .pkc): pawna's system HDF5 instead of the pegasus paths in the Makefile
HDF5_INC=${HDF5_INC:-/usr/include/hdf5/serial}; HDF5_LIB=${HDF5_LIB:-/usr/lib/x86_64-linux-gnu/hdf5/serial}
(cd "$CODES/rockstar" && make with_hdf5_pkc ZSTD_INC="$ZSRC" ZSTD_LIB="$ZBLD" \
   HDF5_FLAGS="-DH5_USE_16_API -DENABLE_HDF5 -I$HDF5_INC -L$HDF5_LIB -lhdf5" > build.log 2>&1) || { tail "$CODES/rockstar/build.log"; exit 1; }
ls -l "$CODES/rockstar/rockstar"

echo "== consistent-trees"
[ -d "$CODES/ctrees/.git" ] || git clone -q https://bitbucket.org/pbehroozi/consistent-trees.git "$CODES/ctrees"
(cd "$CODES/ctrees" && git checkout -q -f 7c20add8bb417c567c8c0725eb38960869b75106 && git clean -qfdx)
overlay "$SB/code/ConsistentTrees/consistent-trees-7c20add" "$CODES/ctrees"
(cd "$CODES/ctrees" && make > build.log 2>&1) || { tail "$CODES/ctrees/build.log"; exit 1; }
ls "$CODES/ctrees/find_parents_and_cleanup" "$CODES/ctrees/assemble_halo_trees" "$CODES/ctrees/gravitational_consistency" > /dev/null

echo "== Python venv"
[ -x "$CODES/venv/bin/python" ] || python3 -m venv --system-site-packages "$CODES/venv"
"$CODES/venv/bin/python" -m pip install -q --upgrade pip
"$CODES/venv/bin/python" -m pip install -q numpy scipy pandas "h5py<3.12" fitsio zstandard
"$CODES/venv/bin/python" -c "import numpy, scipy, pandas, h5py, fitsio, zstandard; print('python', numpy.__version__, scipy.__version__, pandas.__version__, fitsio.__version__)"

echo "== post-process C helpers"
for f in cic_update ngp_update; do cc -O2 -shared -fPIC -o "$CODES/lib/$f.so" "$SB/scripts/post-process/$f.c"; done
ls -l "$CODES/lib"
echo "setup done: CODES=$CODES (put it in the run config)"
