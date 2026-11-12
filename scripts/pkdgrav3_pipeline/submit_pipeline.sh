#!/bin/bash
# Submit the Sahyadri pkdgrav3 pipeline for one run as a chain of PBS jobs:
#   sim -> rockstar -> ctrees -> vahc -> fits     (each starts only if the previous one succeeded: afterok)
#   optional, only when named in --stages: -> xcorr (halo-halo potential cross-correlation, job_xcorr.pbs)
#                                          -> ximult (correlation multipoles, real/redshift space, job_ximult.pbs)
#
# usage: submit_pipeline.sh <run.conf> [--stages sim,rockstar,ctrees,vahc,fits] [--after JOBID] [--hold] [--restart CHK] [--dry-run]
#   --stages  any subset, in pipeline order (e.g. "ctrees,vahc,fits" to redo the later stages)
#   --after   make the first stage wait for an existing job (afterok), e.g. a running simulation
#   --hold    submit the first job held (qrls <id> to start the chain)
#   --restart CHK  the sim stage continues from checkpoint CHK (latest = newest $SIM_DIR/checkpoints/*.chk)
#   --dry-run print the qsub commands only
# Prints the job ids; the chain is also written to $LOGS/chain.<run>.<date>.txt.
set -eu
[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 1; }
CONF=$(readlink -f "$1"); shift
STAGES=sim,rockstar,ctrees,vahc,fits; AFTER=; HOLD=; DRY=; RESTART=
while [ $# -gt 0 ]; do
  case $1 in
    --stages) STAGES=$2; shift 2 ;;
    --after) AFTER=$2; shift 2 ;;
    --hold) HOLD="-h"; shift ;;
    --restart) RESTART=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
HERE=$(dirname "$(readlink -f "$0")")
set -a; source "$CONF"; set +a
[ "$NPART" -le 512 ] || echo "WARNING: NPART=$NPART > 512: runs above 512^3 need approval before submission"
mkdir -p "$LOGS"
QSUB=$(command -v qsub || echo /opt/pbs/bin/qsub)
RUN=$(echo "$STEM" | tr / _)_r$REAL
CHAIN=$LOGS/chain.$RUN.$(date +%Y%m%d_%H%M%S).txt
prev=$AFTER
for st in sim rockstar ctrees vahc fits xcorr ximult; do
  case ",$STAGES," in *",$st,"*) ;; *) continue ;; esac
  case $st in
    sim)      R="select=1:ncpus=$SIM_NCPU:mem=$SIM_MEM";   W=$SIM_WALL ;;
    rockstar) R="select=1:ncpus=$RS_NCPU:mem=$RS_MEM";     W=$RS_WALL ;;
    ctrees)   R="select=1:ncpus=$CT_NCPU:mem=$CT_MEM";     W=$CT_WALL ;;
    vahc)     R="select=1:ncpus=$PP_NCPU:mem=$PP_MEM";     W=$PP_WALL ;;
    fits)     R="select=1:ncpus=$FITS_NCPU:mem=$FITS_MEM"; W=$FITS_WALL ;;
    xcorr)    R="select=1:ncpus=${XC_NCPU:-16}:mem=${XC_MEM:-16gb}"; W=${XC_WALL:-02:00:00} ;;
    ximult)   R="select=1:ncpus=${XM_NCPU:-32}:mem=${XM_MEM:-32gb}"; W=${XM_WALL:-06:00:00} ;;
  esac
  DEP=; [ -n "$prev" ] && DEP="-W depend=afterok:$prev"
  H=; [ -z "$prev" ] && H=$HOLD
  V="CONF=$CONF,PIPE_ENV=$HERE/pipeline_env.sh,STAGE=$st"
  [ "$st" = sim ] && [ -n "$RESTART" ] && V="$V,RESTART=$RESTART"
  cmd=("$QSUB" -N "${st}_$REAL" -q workq -l "$R" -l "walltime=$W" -j oe -o "$LOGS/pbs.$st.$RUN.out" $DEP $H
       -v "$V" "$HERE/job_$st.pbs")
  if [ -n "$DRY" ]; then echo "${cmd[*]}"; prev="<$st>"; continue; fi
  id=$("${cmd[@]}")
  echo "$st $id ($R, $W${DEP:+, $DEP})" | tee -a "$CHAIN"
  prev=$id
done
[ -n "$DRY" ] || echo "logs: $LOGS/<stage>.$RUN.<job>.log; chain: $CHAIN"
