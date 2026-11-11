# Sourced by every pipeline job: reads the run config ($CONF) and sets derived paths.
[ -n "${CONF:-}" ] && [ -f "$CONF" ] || { echo "CONF (run config file) not set or missing: '${CONF:-}'"; exit 2; }
set -a; source "$CONF"; set +a          # export, for make_pkdgrav3_par.py
: "${FORCE_RES:=}" "${RS_PKC_SUBSAMPLES:=}" "${EXTRA_PAR:=}" "${OMEGA_DE:=0}" "${W0:=-1}" "${WA:=0}"
SIM_DIR=$SCRATCH/sims/$STEM/r$REAL          # pkdgrav3 run directory: p.par, snap.<step>.s<pct>.<w>.pkc, snapnames.txt
HALO_DIR=$SCRATCH/halos/$STEM/r$REAL        # Rockstar OUTBASE; later out_<snap>.trees, trees/, compressed/
PIPE=$SANDBOX/scripts/pkdgrav3_pipeline
PP=$SANDBOX/scripts/post-process
PY=$CODES/venv/bin/python
export SAHYADRI_SCRATCH=$SCRATCH SAHYADRI_LIBDIR=$CODES/lib
[ -z "$FORCE_RES" ] && export FORCE_RES=$(awk -v l=$LBOX -v n=$NPART 'BEGIN{printf "%.6g", l/n/50}')
mkdir -p "$LOGS"
# own log file (PBS -o copy-back is unreliable on pawna): $LOGS/<stage>.<run>.<jobid>.log
[ -n "${PBS_JOBID:-}" ] && exec > "$LOGS/${STAGE:-job}.$(echo $STEM | tr / _)_r$REAL.${PBS_JOBID%%.*}.log" 2>&1
export CONF
stamp() { echo "[$(date '+%F %T')] $*"; }
stamp "job ${PBS_JOBID:-interactive} on $(hostname): stage ${STAGE:-?} for $STEM r$REAL (config $CONF)"
# memory of this job (all its processes): $LOGS/<stage>.<run>.<jobid>.mem.tsv, peak reported at exit
if [ -n "${PBS_JOBID:-}" ] || [ "${MEMMON:-0}" = 1 ]; then
  MEMLOG=$LOGS/${STAGE:-job}.$(echo $STEM | tr / _)_r$REAL.${PBS_JOBID:-$$}.mem.tsv; MEMLOG=${MEMLOG/.pawna./.}
  # detached (not a child of this shell), so that a bare `wait` in a job does not wait for the monitor
  MEMMON_PID=$(bash "$PIPE/memmon.sh" $$ "$MEMLOG" "${MEMMON_DT:-5}" < /dev/null > /dev/null 2>&1 & echo $!)
  memmon_report() {
    kill $MEMMON_PID 2>/dev/null
    awk '!/^#/ && NF==4 {if($3>p){p=$3;t=$1;n=$2} if($4>r)r=$4; T=$1}
         END{printf "peak memory: PSS %.2f GB (%d processes, at %.0f s), RSS sum %.2f GB; %d s sampled\n", p/1048576, n, t, r/1048576, T}' "$MEMLOG" |
      while read -r l; do stamp "$l (samples: $MEMLOG)"; done
  }
  trap memmon_report EXIT
else
  memmon_report() { :; }
fi
