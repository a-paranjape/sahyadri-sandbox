#!/bin/bash
# Tests of the Sahyadri cosmology patch of consistent-trees (see ../README_sahyadri.txt).
# usage: run_tests.sh <unmodified consistent-trees 7c20add tree> <scratch dir> [rockstar OUTBASE with out_*.list, rockstar.cfg]
#  1. C and Perl time tables / delta_vir: flat LCDM byte-identical to the original; numerical path vs
#     closed form; C vs Perl for w0-wa.
#  2. (with a Rockstar OUTBASE) full runs: original vs patched for LCDM must give byte-identical trees,
#     hlists and cleaned catalogs; patched with w0 = -0.9, wa = 0.1 is reported.
set -u
ORIG=${1:?unmodified consistent-trees tree}; W=${2:?scratch dir}; RS=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd); PATCH=$(dirname "$HERE")
GEN=$(cd "$PATCH/../../Rockstar/gfcstanford-rockstar-36ce9eea36ee/scripts" && pwd)/gen_merger_cfg.pl
CF="-m64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -D_BSD_SOURCE -D_POSIX_SOURCE -D_POSIX_C_SOURCE=200809L -D_SVID_SOURCE -D_DARWIN_C_SOURCE -Wall"
fails=0; ok() { echo "PASS $1"; }; bad() { echo "FAIL $1"; fails=$((fails+1)); }
rm -rf "$W"; mkdir -p "$W/orig" "$W/new"
cp -R "$ORIG"/. "$W/orig/"; cp -R "$ORIG"/. "$W/new/"
(cd "$PATCH" && find . -type f ! -name "*.bak" ! -path "./tests_sahyadri/*" ! -name README_sahyadri.txt | while read -r f; do cp "$f" "$W/new/$f"; done)

# 1. unit tests
cc -O2 -std=c99 -I"$W/orig/src" "$HERE/time_table_test.c" "$W/orig/src/universe_time.c" "$W/orig/src/masses.c" -o "$W/t_old" -lm
cc -O2 -std=c99 -DNEWCODE -I"$W/new/src" "$HERE/time_table_test.c" "$W/new/src/universe_time.c" "$W/new/src/masses.c" -o "$W/t_new" -lm
"$W/t_old" 0 > "$W/c_old"; for m in 0 1 2 3; do "$W/t_new" $m > "$W/c_new$m"; done
cmp -s "$W/c_old" "$W/c_new0" && ok "C: no dark-energy call == original" || bad "C: no dark-energy call == original"
cmp -s "$W/c_old" "$W/c_new3" && ok "C: explicit flat LCDM == original" || bad "C: explicit flat LCDM == original"
perl -I"$W/orig/src" "$HERE/time_table_test.pl" 0 > "$W/p_old" 2>/dev/null; for m in 0 1 2 3; do perl -I"$W/new/src" "$HERE/time_table_test.pl" $m > "$W/p_new$m" 2>/dev/null; done
cmp -s "$W/p_old" "$W/p_new0" && ok "Perl: init(Om, h) == original" || bad "Perl: init(Om, h) == original"
cmp -s "$W/p_old" "$W/p_new3" && ok "Perl: explicit flat LCDM == original" || bad "Perl: explicit flat LCDM == original"
r=$(paste "$W/c_old" "$W/c_new1" | awk '$2!=0 {d=($2-$5)/$2; if(d<0)d=-d; if(d>m)m=d} END {print m+0}')
awk -v r="$r" 'BEGIN {exit !(r < 1e-6)}' && ok "C: numerical vs closed-form time table, max rel. diff $r" || bad "C: numerical vs closed form ($r)"
r=$(paste "$W/c_new2" "$W/p_new2" | awk '$2!=0 {d=($2-$5)/$2; if(d<0)d=-d; if(d>m)m=d} END {print m+0}')
awk -v r="$r" 'BEGIN {exit !(r < 1e-7)}' && ok "C vs Perl, w0-wa time table, max rel. diff $r" || bad "C vs Perl w0-wa ($r)"

# 2. full runs
if [ -n "$RS" ]; then
  for b in orig new; do (cd "$W/$b/src" && make CFLAGS="$CF" > ../build.log 2>&1) || bad "build $b"; done
  run() { local name=$1 code=$2 de=$3 R=$W/run_$1
    mkdir -p "$R"; cp "$RS"/out_*.list "$R/"
    sed "s|^OUTBASE = .*|OUTBASE = \"$R\"|" "$RS/rockstar.cfg" > "$R/rockstar.cfg"
    (cd "$R" && perl "$GEN" rockstar.cfg > gen.log 2>&1)
    [ "$de" = w0wa ] && sed -i.orig 's/^w0=.*/w0=-0.9/; s/^wa=.*/wa=0.1/' "$R/outputs/merger_tree.cfg"
    (cd "$code" && perl do_merger_tree.pl "$R/outputs/merger_tree.cfg" "$code" > "$R/trees.log" 2>&1 &&
                   perl halo_trees_to_catalog.pl "$R/outputs/merger_tree.cfg" "$code" > "$R/cat.log" 2>&1) || bad "run $name"; }
  run A "$W/orig" lcdm; run B "$W/new" lcdm; run C "$W/new" w0wa
  n=0; d=0; for f in $(cd "$W/run_A" && ls trees/* hlists/* outputs/*.list); do n=$((n+1)); cmp -s "$W/run_A/$f" "$W/run_B/$f" || d=$((d+1)); done
  [ $d = 0 ] && ok "LCDM full run: $n output files byte-identical (original vs patched)" || bad "LCDM full run: $d of $n files differ"
  d=0; for f in $(cd "$W/run_A" && ls trees/* hlists/* outputs/*.list); do cmp -s "$W/run_A/$f" "$W/run_C/$f" || d=$((d+1)); done
  echo "INFO w0 = -0.9, wa = 0.1 on the same catalogs: $d of $n files change"
fi
echo "== $fails failure(s)"; exit $((fails > 0))
