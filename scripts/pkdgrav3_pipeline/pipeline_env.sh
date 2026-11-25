# Sourced by every pipeline job: reads the run config ($CONF) and sets derived paths.
[ -n "${CONF:-}" ] && [ -f "$CONF" ] || { echo "CONF (run config file) not set or missing: '${CONF:-}'"; exit 2; }
set -a; source "$CONF"; set +a          # export, for make_pkdgrav3_par.py
: "${FORCE_RES:=}" "${RS_PKC_SUBSAMPLES:=}" "${EXTRA_PAR:=}" "${PKC_SUBDIR:=}" "${PKD_PAR:=}" "${RS_CFG:=}" "${OMEGA_DE:=0}" "${W0:=-1}" "${WA:=0}"
SIM_DIR=$SCRATCH/sims/$STEM/r$REAL          # pkdgrav3 run directory: p.par, snapnames.txt, logs
PKC_DIR=$SIM_DIR${PKC_SUBDIR:+/$PKC_SUBDIR}  # the snap.<step>.s<pct>.<w>.pkc files (PKC_SUBDIR, e.g. compressed)
HALO_DIR=$SCRATCH/halos/$STEM/r$REAL        # Rockstar OUTBASE; later out_<snap>.trees, trees/, compressed/
PIPE=$SANDBOX/scripts/pkdgrav3_pipeline
PP=$SANDBOX/scripts/post-process
PY=$CODES/venv/bin/python
export SAHYADRI_SCRATCH=$SCRATCH SAHYADRI_LIBDIR=$CODES/lib
[ -d "$CODES/bin" ] && export PATH=$CODES/bin:$PATH   # tools built for the pipeline (e.g. pigz on swarm)
[ -z "$FORCE_RES" ] && export FORCE_RES=$(awk -v l=$LBOX -v n=$NPART 'BEGIN{printf "%.6g", l/n/50}')
mkdir -p "$LOGS"
# own log file (PBS -o copy-back is unreliable on pawna): $LOGS/<stage>.<run>.<jobid>.log
[ -n "${PBS_JOBID:-}" ] && exec > "$LOGS/${STAGE:-job}.$(echo $STEM | tr / _)_r$REAL.${PBS_JOBID%%.*}.log" 2>&1
export CONF
stamp() { echo "[$(date '+%F %T')] $*"; }
stamp "job ${PBS_JOBID:-interactive} on $(hostname): stage ${STAGE:-?} for $STEM r$REAL (config $CONF)"
# ---------------------------------------------------------------- events (for tools/sahyadri_manager)
# Every job writes its own file $STATE_DIR/events/<date>.<stage>.<job>.jsonl, one JSON object per line
# (one writer per file: appends from several NFS clients to one file are not safe). The manager on the head node
# watches $STATE_DIR with inotifywait. Lines: {"t": epoch, "time": ISO, "run", "stage", "job", "host", "ev", ...}.
#   event <ev> [key=value ...]     values are written as JSON strings unless they are plain integers
STATE_DIR=$SCRATCH/state/$STEM/r$REAL
_EVJOB=${PBS_JOBID%%.*}; _EVJOB=${_EVJOB:-pid$$}
mkdir -p "$STATE_DIR/events" 2>/dev/null
EVFILE=$STATE_DIR/events/$(date +%Y%m%dT%H%M%S).${STAGE:-job}.$_EVJOB.jsonl
_json_str() { local s=${1//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\n'/ }; s=${s//$'\t'/ }; printf '"%s"' "$s"; }
event() {
  local ev=$1 kv k v line; shift
  line=$(printf '{"t": %s, "time": "%s", "run": %s, "stage": %s, "job": %s, "host": %s, "ev": %s' "$(date +%s)" \
    "$(date -Iseconds)" "$(_json_str "$STEM/r$REAL")" "$(_json_str "${STAGE:-job}")" "$(_json_str "$_EVJOB")" \
    "$(_json_str "$(hostname -s)")" "$(_json_str "$ev")")
  for kv in "$@"; do
    k=${kv%%=*}; v=${kv#*=}
    if [[ $v =~ ^-?(0|[1-9][0-9]{0,14})$ ]]; then line="$line, $(_json_str "$k"): $v"; else line="$line, $(_json_str "$k"): $(_json_str "$v")"; fi
  done
  printf '%s}\n' "$line" >> "$EVFILE" 2>/dev/null
  return 0
}
event start conf="$CONF" log="$LOGS/${STAGE:-job}.$(echo $STEM | tr / _)_r$REAL.$_EVJOB.log" ${RS_SNAP_FIRST:+first=$RS_SNAP_FIRST} ${RS_SNAP_LAST:+last=$RS_SNAP_LAST}
# the end event is written by the EXIT trap (also on qdel / wall time: PBS sends SIGTERM first)
_on_exit() { local rc=$?; eval "${EXIT_HOOK:-:}"; memmon_report; event end rc=$rc; }   # EXIT_HOOK: a job's own clean-up
trap 'exit 143' TERM
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
else
  memmon_report() { :; }
fi
trap _on_exit EXIT
# ---------------------------------------------------------------- node check before the job's work (node_check.sh)
# A job on a node whose NFS client hangs, whose memory is held by unreclaimable cache or whose local disk is full
# stalls for hours (pawna node1, 23-25 Nov 2026). PBS jobs check the node first: NODE_CHECK=1 (default where SITE is
# unset or pawna; 0 = off). On a bad node: event node_bad (host, reasons) and exit 75 before anything is touched;
# the run manager marks the node bad and resubmits the job on another node (not a failure of the run).
_ncdef=0; [ "${SITE:-pawna}" = pawna ] && _ncdef=1
if [ -n "${PBS_JOBID:-}" ] && [ "${NODE_CHECK:-$_ncdef}" = 1 ] && [ "${STAGE:-}" != archive ] && [ "${STAGE:-}" != archb ]; then
  _gb() { case $1 in *gb) echo "${1%gb}" ;; *tb) echo $(( ${1%tb} * 1024 )) ;; *) echo 0 ;; esac; }
  _ncm=0; _nct=0; _ncw=0
  case ${STAGE:-} in
    sim)      _ncm=$(_gb "${SIM_MEM:-0}")
              # staged checkpoint (~45 B/particle measured: 385 GB at 2048^3) plus a snapshot
              [ "${SIM_STAGING:-0}" = 1 ] && { _nct=$(awk -v n="$NPART" 'BEGIN{printf "%.0f", n^3 * 50 / 1e9}'); _ncw=$_nct; } ;;
    rockstar) _ncm=$(_gb "${RS_MEM:-0}")
              # local output ~65 GB per 2048^3 snapshot (r3/r4 batches, 25 Nov); bad below half of the estimate
              if [ "${RS_LOCAL_OUT:-1}" = 1 ] && [ "${RS_NODES:-1}" = 1 ]; then
                _k=$(( ${RS_SNAP_LAST:-0} - ${RS_SNAP_FIRST:-0} + 1 ))
                _ncw=$(awk -v n="$NPART" -v k="$_k" 'BEGIN{printf "%.0f", 65 * k * (n / 2048)^3}'); _nct=$((_ncw / 2))
              fi ;;
    ctrees)   _ncm=$(_gb "${CT_MEM:-0}") ;;
    vahc|vpot) _ncm=$(_gb "${PP_MEM:-0}") ;;
    fits)     _ncm=$(_gb "${FITS_MEM:-0}") ;;
    verify)   _ncm=$(_gb "${VF_MEM:-0}") ;;
  esac
  _ncm=$(( _ncm * 9 / 10 ))      # MemAvailable at least 90 % of the request
  _ncout=$(bash "$PIPE/node_check.sh" --paths "$STATE_DIR" --mem "$_ncm" --tmp "$_nct" --tmp-warn "$_ncw" 2>&1)
  _ncrc=$?
  echo "$_ncout" | grep -v '^NODECHECK' | while read -r l; do stamp "$l"; done
  if [ $_ncrc = 3 ]; then
    _ncr=$(echo "$_ncout" | grep ' BAD ' | sed 's/^node_check [^ ]* BAD  //' | paste -sd ';' -)
    event node_bad reasons="$_ncr"
    stamp "node $(hostname -s) failed the node check: $_ncr; exit 75 (the manager resubmits on another node)"
    exit 75
  fi
fi
# A simulation job without RESTART starts from the ICs. Refuse that when the run already has checkpoints or
# snapshots: e.g. PBS re-running a first job after a node or power failure (jobs are rerunnable by default) would
# otherwise write over the run's outputs and rotate away its good checkpoints. Restart from a checkpoint instead
# (manager: "Restart from checkpoint"; submit_pipeline.sh --restart latest); SIM_FROM_IC=1 to really start again.
if [ "${STAGE:-}" = sim ] && [ -z "${RESTART:-}" ] && [ "${SIM_FROM_IC:-0}" != 1 ]; then
  _nchk=$(ls "$SIM_DIR"/checkpoints/*.chk 2>/dev/null | wc -l)
  _nsnap=$(ls "$PKC_DIR"/snap.*.pkc 2>/dev/null | head -1 | wc -l)
  if [ "$_nchk" -gt 0 ] || [ "$_nsnap" -gt 0 ]; then
    stamp "refusing to start the simulation from the ICs: $SIM_DIR has $_nchk checkpoint(s) and snapshots ($_nsnap); restart from a checkpoint (RESTART=latest) or set SIM_FROM_IC=1"
    event sim_refused reason="checkpoints or snapshots exist; not started from the ICs" nchk="$_nchk"
    exit 3
  fi
fi
