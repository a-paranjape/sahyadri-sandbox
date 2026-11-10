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
