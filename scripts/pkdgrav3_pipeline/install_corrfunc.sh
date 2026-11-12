#!/bin/bash
# Install Corrfunc from Arnaud de Mattia's fork (github.com/adematti/Corrfunc, branch desi: signed mu, i.e. the odd
# multipoles) into a Python environment, for scripts/post-process/halo_xi_multipoles.py (stage ximult).
#
# usage: install_corrfunc.sh <python> [workdir]
#   python   the venv's python (e.g. $CODES/venv/bin/python); numpy must be installed in it
#   workdir  where the source is cloned (default: a temporary directory, removed afterwards)
# Environment: CORRFUNC_REPO (default https://github.com/adematti/Corrfunc), CORRFUNC_REF (default desi),
#              LIBOMP_PREFIX (Mac; default $(brew --prefix libomp) or /opt/homebrew/opt/libomp)
#
# Two patches of Corrfunc's common.mk, applied only where they are needed:
#  - Linux: numpy's include directory first. Corrfunc puts -isystem <python include> before numpy's, and on
#    Debian/Ubuntu with the python3-numpy package <python include>/numpy is the *system* numpy (1.x) headers, so a
#    venv with numpy 2 got an extension that fails with "numpy.core.multiarray failed to import" (pawna, 2026-11-12).
#  - Mac: Apple clang has no OpenMP and Homebrew gcc rejects Corrfunc's -march=apple-m1, so OpenMP comes from
#    Homebrew's libomp (brew install libomp) with clang; without it the build is serial (about 7x slower on 8 cores).
# The build uses this python's numpy (--no-build-isolation; needs setuptools and wheel, installed here).
# Finally a smoke test: signed mu bins from -1 to 1, and whether the extension is linked to OpenMP.
set -eu
PY=${1:?usage: install_corrfunc.sh <python> [workdir]}
# absolute path, but not resolved: a venv's bin/python is a symlink and must be called through the venv path
case $PY in */*) PY=$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY") ;; *) PY=$(command -v "$PY") ;; esac
REPO=${CORRFUNC_REPO:-https://github.com/adematti/Corrfunc}
REF=${CORRFUNC_REF:-desi}
"$PY" -c "import numpy" || { echo "numpy is not installed in $PY"; exit 1; }
if [ $# -ge 2 ]; then W=$2; mkdir -p "$W"; else W=$(mktemp -d); trap 'rm -rf "$W"' EXIT; fi
rm -rf "$W/Corrfunc"
git clone -q -b "$REF" --depth 1 "$REPO" "$W/Corrfunc"
cd "$W/Corrfunc"
echo "Corrfunc $(git log --oneline -1) from $REPO@$REF"
MK=common.mk
grep -q '^ *export PYTHON_CFLAGS := $(PYTHON_INCL) $(NUMPY_INCL_FLAG)' $MK &&
  sed -i.orig 's/^\( *export PYTHON_CFLAGS := \)$(PYTHON_INCL) $(NUMPY_INCL_FLAG)/\1$(NUMPY_INCL_FLAG) $(PYTHON_INCL)/' $MK &&
  echo "patched: numpy include before the python include"
if [ "$(uname)" = Darwin ]; then
  P=${LIBOMP_PREFIX:-$(brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)}
  if [ -f "$P/lib/libomp.dylib" ]; then
    "$PY" - "$MK" "$P" <<'EOF'
import sys
mk, p = sys.argv[1], sys.argv[2]
s = open(mk).read()
o = """          # Apple clang/gcc does not support OpenMP
          ifeq (Apple, $(findstring Apple, $(CC_VERSION)))"""
n = """          # Apple clang/gcc does not support OpenMP (unless Homebrew's libomp is used)
          ifneq ($(wildcard %s/lib/libomp.dylib),)
            CLANG_OMP_AVAIL := true
            CFLAGS += -Xpreprocessor -fopenmp -I%s/include
            CLINK  += -L%s/lib -lomp -Wl,-rpath,%s/lib
          else ifeq (Apple, $(findstring Apple, $(CC_VERSION)))""" % (p, p, p, p)
if o not in s:
    sys.exit('common.mk: the Apple clang block was not found; not patched (serial build)')
open(mk, 'w').write(s.replace(o, n, 1))
print('patched: OpenMP from', p)
EOF
  else
    echo "WARNING: no libomp in $P (brew install libomp): Corrfunc will be built without OpenMP"
  fi
fi
"$PY" -m pip install -q setuptools wheel
"$PY" -m pip install -q --no-build-isolation --force-reinstall --no-deps .
cd /
"$PY" - <<'EOF'
import glob, os, subprocess, sys
import numpy as np
import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
x = np.random.default_rng(0).random((3, 2000)) * 200.0
r = DDsmu(1, 2, np.linspace(0, 100, 11), 1.0, 10, *x, boxsize=200.0, periodic=True)
lo, hi = r['mumax'].min(), r['mumax'].max()
so = glob.glob(os.path.join(os.path.dirname(Corrfunc.__file__), '_countpairs.*'))[0]
tool = ['otool', '-L'] if sys.platform == 'darwin' else ['ldd']
omp = [os.path.basename(l.split()[0]) for l in subprocess.run(tool + [so], capture_output=True, text=True).stdout.splitlines()
       if any(k in l for k in ('libomp', 'libgomp', 'libiomp'))]
print('Corrfunc %s (numpy %s): %d pairs, mu bin upper edges %.2f .. %.2f (signed: %s), OpenMP: %s'
      % (Corrfunc.__version__, np.__version__, r['npairs'].sum(), lo, hi, lo < 0, omp or 'no'))
if not lo < 0:
    sys.exit('this Corrfunc counts |mu| only: not the desi fork')
EOF
