#!/bin/bash
# Submit the whole Gadget-4 vs pkdgrav3 comparison as a chain of PBS jobs:
#   gadget (CLASS, NGenIC ICs, IC conversion, Gadget-4 run)
#     -> pkdgrav3 from the same ICs: sim -> rockstar -> ctrees      (pkdgrav3_pipeline, compare32.conf)
#     -> Gadget-4 halos:             rockstar -> ctrees             (pkdgrav3_pipeline, compare32_gadget.conf)
#   -> compare (after both ctrees jobs)
# usage: submit_compare.sh <compare32.conf> [--dry-run] [--hold]
#   the Gadget-4 config is <dir of conf>/<name>_gadget.conf
set -eu
[ $# -ge 1 ] || { sed -n '2,9p' "$0"; exit 1; }
CONF=$(readlink -f "$1"); shift
GCONF=${CONF%.conf}_gadget.conf
[ -f "$GCONF" ] || { echo "missing $GCONF"; exit 1; }
DRY=; HOLD=
for o in "$@"; do case $o in --dry-run) DRY=1 ;; --hold) HOLD="-h" ;; *) echo "unknown option $o"; exit 1 ;; esac; done
HERE=$(dirname "$(readlink -f "$0")")
set -a; source "$CONF"; set +a
PIPE=$SANDBOX/scripts/pkdgrav3_pipeline
[ "$NPART" -le 512 ] || { echo "NPART=$NPART > 512 needs approval"; exit 1; }
mkdir -p "$LOGS"
QSUB=$(command -v qsub || echo /opt/pbs/bin/qsub)
RUN=pkd${NPART}_vs_gadget${NPART}_r$REAL
CHAIN=$LOGS/chain.$RUN.$(date +%Y%m%d_%H%M%S).txt
run() { if [ -n "$DRY" ]; then echo "$*" >&2; echo "<$1>"; else "$@"; fi; }

G=$(run "$QSUB" -N g4_$REAL -q workq -l select=1:ncpus=$G4_NCPU:mpiprocs=$G4_NCPU:mem=$G4_MEM -l walltime=$G4_WALL \
      -j oe -o "$LOGS/pbs.gadget.$RUN.out" $HOLD -v "CONF=$GCONF,PIPE_ENV=$PIPE/pipeline_env.sh,STAGE=gadget" \
      "$HERE/job_gadget.pbs")
echo "gadget $G" | tee -a "$CHAIN"
D=; [ -n "$DRY" ] && D=--dry-run
P=$("$PIPE/submit_pipeline.sh" "$CONF" --stages sim,rockstar,ctrees --after "$G" $D 2>&1 | tee -a "$CHAIN")
echo "$P"
H=$("$PIPE/submit_pipeline.sh" "$GCONF" --stages rockstar,ctrees --after "$G" $D 2>&1 | tee -a "$CHAIN")
echo "$H"
if [ -n "$DRY" ]; then
  echo "$QSUB -N cmp_$REAL ... -W depend=afterok:<ctrees pkdgrav3>:<ctrees gadget> $HERE/job_compare.pbs"; exit 0
fi
C1=$(echo "$P" | awk '$1=="ctrees"{print $2}'); C2=$(echo "$H" | awk '$1=="ctrees"{print $2}')
[ -n "$C1" ] && [ -n "$C2" ] || { echo "could not find the ctrees job ids"; exit 1; }
C=$("$QSUB" -N cmp_$REAL -q workq -l select=1:ncpus=$CMP_NCPU:mem=$CMP_MEM -l walltime=$CMP_WALL -j oe \
      -o "$LOGS/pbs.compare.$RUN.out" -W depend=afterok:$C1:$C2 \
      -v "CONF=$CONF,PIPE_ENV=$PIPE/pipeline_env.sh,STAGE=compare" "$HERE/job_compare.pbs")
echo "compare $C (after $C1 $C2)" | tee -a "$CHAIN"
echo "chain: $CHAIN; results: $CMP_DIR"
