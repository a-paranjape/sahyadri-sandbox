#!/bin/bash -l
# Build the two Gadget-4 executables of the Sahyadri workflow for one particle number, in the directory layout that
# scripts/gadget/run_gadget.sh expects:
#   <G4HOME>/code/Gadget-4/mesh<2N>-NGenIC/Gadget4   creates the ICs (RestartFlag 6)
#   <G4HOME>/code/Gadget-4/mesh<2N>/Gadget4          runs from the IC file
# (run_gadget.sh names the directory after NMESH = 2 * NPART = PMGRID.)
#
# usage: setup_gadget4_pawna.sh <gadget4 source (git clone)> <G4HOME> <NPART>
# The source is copied to <G4HOME>/code/Gadget-4/gadget4 first, so nothing is written into it.
# Toolchain: the pawna login environment (Intel oneAPI MPI mpicxx around GCC, /apps/{gsl,fftw3,hdf5}), SYSTYPE
# "pawna" as in the existing Sahyadri builds.
set -eu
SRC=$(readlink -f "$1"); G4HOME=$(readlink -m "$2"); N=$3
HERE=$(dirname "$(readlink -f "$0")")
D=$G4HOME/code/Gadget-4
mkdir -p "$D"
if [ ! -d "$D/gadget4" ]; then
  cp -a "$SRC" "$D/gadget4"
fi
cd "$D/gadget4"
echo "Gadget-4 source: $(git log -1 --format='%h %ad %s' --date=short 2>/dev/null || echo 'not a git checkout')"
echo 'SYSTYPE="pawna"' > Makefile.systype
export LD_LIBRARY_PATH=/apps/gsl/lib:/apps/hdf5/lib:/apps/fftw3/lib:${LD_LIBRARY_PATH:-}
echo "mpicxx: $(command -v mpicxx)"
for kind in NGenIC run; do
  if [ $kind = NGenIC ]; then B=$D/mesh$((2*N))-NGenIC; else B=$D/mesh$((2*N)); fi
  mkdir -p "$B"
  sed -e "s/@PMGRID@/$((2*N))/; s/@NGENIC@/$N/" "$HERE/Config.sh.template" > "$B/Config.sh"
  if [ $kind = run ]; then sed -i -e '/^NGENIC/d; /^CREATE_GRID/d' "$B/Config.sh"; fi
  echo "== $B"; grep -v '^#' "$B/Config.sh"
  make -j 16 DIR="$B" > "$B/make.log" 2>&1 || { tail -30 "$B/make.log"; exit 1; }
  ls -la "$B/Gadget4"
done
