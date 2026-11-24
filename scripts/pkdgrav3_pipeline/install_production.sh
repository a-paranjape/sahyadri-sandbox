#!/bin/bash -l
# Production install of everything the Sahyadri pipeline needs on one site, in a new directory:
#   <root>/pkdgrav3_io        pkdgrav3 (GitHub main) + snapcomp tools; pawna: own venv for pkdgrav3's embedded
#                             Python (dill, tomli, Cython); pegasus, swarm: conda env (MPI, HDF5, FFTW)
#   <root>/sahyadri-sandbox   this repository (branch run-manager): pipeline, post-processing, run manager
#   <root>/codes              Rockstar (.pkc reader), consistent-trees, Python venv (requirements.txt), C helpers
#
# usage: install_production.sh pawna   [/user/$USER/sahyadri_prod]
#        install_production.sh pegasus [~/sahyadri_pipeline]
#        install_production.sh swarm   [/user1/$USER/sahyadri_pipeline]   (Miniforge in /user1/$USER/miniforge3 if
#                                       missing; pkdgrav3 for Ice Lake; no 'bash -l': swarm's login shell drops the
#                                       arguments of a script started that way)
#        install_production.sh pride   [/user1/$USER/sahyadri_pride]      (TIFR; shares swarm's Lustre /user1 and
#                                       Miniforge; pkdgrav3 for Haswell (AVX2): swarm's Ice Lake build does not run)
# Rerunning updates the clones and rebuilds. Nothing outside <root> is changed (pegasus: caches on /data).
set -eu
SITE=${1:?usage: install_production.sh pawna|pegasus|swarm|pride [root]}
case $SITE in
  pawna)   ROOT=${2:-/user/$USER/sahyadri_prod} ;;
  pegasus) ROOT=${2:-$HOME/sahyadri_pipeline} ;;
  swarm)   ROOT=${2:-/user1/$USER/sahyadri_pipeline} ;;
  pride)   ROOT=${2:-/user1/$USER/sahyadri_pride} ;;
  *) echo "unknown site $SITE"; exit 1 ;;
esac
BRANCH=${BRANCH:-run-manager}
PKD_BRANCH=${PKD_BRANCH:-main}
mkdir -p "$ROOT"; ROOT=$(cd "$ROOT" && pwd)
R=$ROOT/pkdgrav3_io
mkdir -p "$R"
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "== repository ($BRANCH) in $ROOT/sahyadri-sandbox"
if [ -d "$ROOT/sahyadri-sandbox/.git" ]; then
  git -C "$ROOT/sahyadri-sandbox" fetch -q origin && git -C "$ROOT/sahyadri-sandbox" checkout -q "$BRANCH" \
    && git -C "$ROOT/sahyadri-sandbox" merge -q --ff-only "origin/$BRANCH"
else
  git clone -q -b "$BRANCH" https://github.com/a-paranjape/sahyadri-sandbox.git "$ROOT/sahyadri-sandbox"
fi
git -C "$ROOT/sahyadri-sandbox" log -1 --format='sahyadri-sandbox %h %s'

stamp "== pkdgrav3 ($PKD_BRANCH) in $R"
[ -d "$R/pkdgrav3/.git" ] || git clone -q -b "$PKD_BRANCH" https://github.com/shadaba/pkdgrav3.git "$R/pkdgrav3"
git -C "$R/pkdgrav3" fetch -q origin && git -C "$R/pkdgrav3" merge -q --ff-only "origin/$PKD_BRANCH"
git -C "$R/pkdgrav3" log -1 --format='pkdgrav3 %h %s'
if [ "$SITE" = pawna ]; then
  # system python3.10 is embedded in pkdgrav3; its packages come from this venv (env_pawna.sh sets PYTHONPATH)
  [ -x "$R/venv/bin/python" ] || /usr/bin/python3 -m venv "$R/venv"
  "$R/venv/bin/python" -m pip install -q --upgrade pip
  "$R/venv/bin/python" -m pip install -q "numpy<2" "cython>=3" dill tomli
  PKD_IO_ROOT=$R bash -l "$R/pkdgrav3/bench/phase0/compile_pawna.sh"
  ENVSH=$R/pkdgrav3/bench/phase0/env_pawna.sh
elif [ "$SITE" = swarm ] || [ "$SITE" = pride ]; then
  export CONDA_BASE=${CONDA_BASE:-/user1/$USER/miniforge3}
  CACHE=${CACHE:-$ROOT/cache}
  mkdir -p "$CACHE/pip"
  if [ ! -x "$CONDA_BASE/bin/conda" ]; then
    curl -sSL -o "$CACHE/Miniforge3.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$CACHE/Miniforge3.sh" -b -p "$CONDA_BASE"
  fi
  # same conda env and build as pegasus; swarm: Ice Lake (Xeon 8358), pride: Haswell (Xeon E5-2660 v3, no AVX-512)
  ARCH=$([ "$SITE" = swarm ] && echo icelake-server || echo haswell) \
    bash "$R/pkdgrav3/tools/pegasus/setup_pegasus.sh" "$R"
  ENVSH=$ROOT/sahyadri-sandbox/scripts/pkdgrav3_pipeline/$SITE/env_$SITE.sh
else
  CACHE=${CACHE:-/data/project/hpc2502016/shadab/cache}
  mkdir -p "$CACHE/conda_pkgs" "$CACHE/pip"
  [ -e "$R/conda_pkgs" ] || ln -s "$CACHE/conda_pkgs" "$R/conda_pkgs"
  bash -l "$R/pkdgrav3/tools/pegasus/setup_pegasus.sh" "$R"
  ENVSH=$R/pkdgrav3/tools/pegasus/env_pegasus.sh
fi
ls -la "$R/build/pkdgrav3" "$R/build_snapcomp/pkc_check"

stamp "== Rockstar, consistent-trees, Python venv, post-processing helpers in $ROOT/codes"
if [ "$SITE" = pawna ]; then
  bash "$ROOT/sahyadri-sandbox/scripts/pkdgrav3_pipeline/setup_pawna.sh" "$ROOT/codes" "$R"
elif [ "$SITE" = swarm ] || [ "$SITE" = pride ]; then
  PIP_CACHE_DIR=$CACHE/pip bash "$ROOT/sahyadri-sandbox/scripts/pkdgrav3_pipeline/setup_pegasus.sh" "$ROOT/codes" "$R"
else
  PIP_CACHE_DIR=$CACHE/pip bash "$ROOT/sahyadri-sandbox/scripts/pkdgrav3_pipeline/setup_pegasus.sh" "$ROOT/codes" "$R"
fi

stamp "== check"
"$ROOT/codes/venv/bin/python" - "$ROOT/sahyadri-sandbox" <<'EOF'
import importlib, sys
sb = sys.argv[1]
sys.path[:0] = [sb + '/scripts/pkdgrav3_pipeline', sb + '/scripts/post-process']
mods = ['numpy', 'scipy', 'pandas', 'h5py', 'fitsio', 'zstandard', 'matplotlib', 'classy', 'IPython',
        'retention', 'pk_measure', 'class_pk', 'vahc_fast', 'pkcreader', 'readers', 'correlations', 'voronoi_web',
        'halo_potential_xcorr', 'vpot_hdf5', 'ConvertToFITS']
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        bad.append('%s (%s)' % (m, e))
print('imports ok: %d of %d%s' % (len(mods) - len(bad), len(mods), ('; FAILED: ' + '; '.join(bad)) if bad else ''))
sys.exit(1 if bad else 0)
EOF
ls "$ROOT/codes/rockstar/rockstar" "$ROOT/codes/ctrees/find_parents_and_cleanup" "$ROOT/codes/lib"
cat > "$ROOT/README.txt" <<EOF
Sahyadri production install on $SITE, written by install_production.sh $(date '+%F %T')
  sahyadri-sandbox/  repository (branch $BRANCH): $(git -C "$ROOT/sahyadri-sandbox" log -1 --format='%h %s')
  pkdgrav3_io/       pkdgrav3: $(git -C "$R/pkdgrav3" log -1 --format='%h %s'); environment: $ENVSH
  codes/             Rockstar, consistent-trees, venv (scripts/pkdgrav3_pipeline/requirements.txt), lib/*.so
setup_production.py --site $SITE uses these paths by default.
EOF
stamp "done: $ROOT"
