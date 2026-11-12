#!/bin/bash
# One-time setup of the halo codes and Python environment of the Sahyadri pkdgrav3 pipeline on pawna.
#
# usage: setup_pawna.sh <CODES dir> [PKD_IO_ROOT]
#   CODES        where to build (e.g. /storage/$USER/sahyadri_codes); the run configs point to it
#   PKD_IO_ROOT  pkdgrav3 install (default /storage/$USER/pkdgrav3_io; set it up first with pkdgrav3's
#                tools/pawna/setup_pawna.sh, see pkdgrav3 docs/running_on_pawna.md). Its snapcomp build
#                provides the zstd library that Rockstar's .pkc reader links against.
# Builds:
#   $CODES/rockstar   Rockstar from github.com/shadaba/rockstar, branch sahyadri at ROCKSTAR_REF (upstream 36ce9eea36ee +
#                     Sahyadri changes: PKC reader, curvature, w0-wa; see SAHYADRI.md there), make with_hdf5_pkc.
#                     (public). If it cannot be fetched (no network to GitHub), the same source is made from
#                     upstream 36ce9ee plus the older overlay kept here in code/Rockstar/.
#   $CODES/ctrees     consistent-trees 7c20add + Sahyadri changes (w0-wa/curvature background), make
#   $CODES/venv       Python 3 venv (own packages) + numpy scipy pandas h5py fitsio zstandard
#   $CODES/lib        cic_update.so, ngp_update.so, vpot_sums.so, halo_pair_sums.so for post-process (SAHYADRI_LIBDIR)
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
ROCKSTAR_REPO=${ROCKSTAR_REPO:-https://github.com/shadaba/rockstar.git}
ROCKSTAR_REF=${ROCKSTAR_REF:-5a73982879a5d43106cc601ec97aad6dc609c416}   # branch sahyadri
[ -d "$CODES/rockstar/.git" ] || git clone -q https://bitbucket.org/gfcstanford/rockstar.git "$CODES/rockstar"
git -C "$CODES/rockstar" remote get-url sahyadri >/dev/null 2>&1 || git -C "$CODES/rockstar" remote add sahyadri "$ROCKSTAR_REPO"
if GIT_TERMINAL_PROMPT=0 git -C "$CODES/rockstar" fetch -q sahyadri 2>/dev/null; then
  (cd "$CODES/rockstar" && git checkout -q -f "$ROCKSTAR_REF" && git clean -qfdx)
  echo "Rockstar from $ROCKSTAR_REPO at $ROCKSTAR_REF"
else
  echo "cannot fetch $ROCKSTAR_REPO: upstream 36ce9ee + overlay from $SB/code/Rockstar (same source)"
  (cd "$CODES/rockstar" && git checkout -q -f 36ce9eea36eeda4c333acf56f8bb0d40ff0df2a1 && git clean -qfdx)
  overlay "$SB/code/Rockstar/gfcstanford-rockstar-36ce9eea36ee" "$CODES/rockstar"
fi
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
# self-contained (no system site packages): pawna's system scipy is built for numpy < 1.25
[ -x "$CODES/venv/bin/python" ] || python3 -m venv "$CODES/venv"
"$CODES/venv/bin/python" -m pip install -q --upgrade pip
"$CODES/venv/bin/python" -m pip install -q numpy scipy pandas "h5py<3.12" fitsio zstandard
"$CODES/venv/bin/python" -c "import numpy, scipy, pandas, h5py, fitsio, zstandard; print('python', numpy.__version__, scipy.__version__, pandas.__version__, fitsio.__version__)"

echo "== post-process C helpers"
for f in cic_update ngp_update; do cc -O2 -shared -fPIC -o "$CODES/lib/$f.so" "$SB/scripts/post-process/$f.c"; done
cc -O3 -fopenmp -shared -fPIC -o "$CODES/lib/vpot_sums.so" "$SB/scripts/post-process/vpot_sums.c"   # .vpot (vahc_fast.py)
cc -O3 -fopenmp -shared -fPIC -o "$CODES/lib/halo_pair_sums.so" "$SB/scripts/post-process/halo_pair_sums.c"   # halo_potential_xcorr.py
ls -l "$CODES/lib"
echo "setup done: CODES=$CODES (put it in the run config)"
