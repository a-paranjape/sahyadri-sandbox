#!/bin/bash
# Submit the Sahyadri pkdgrav3 pipeline for one run as a chain of PBS jobs:
#   sim -> rockstar -> ctrees -> vahc -> fits -> parquet   (each starts only if the previous one succeeded: afterok;
#   parquet only with PARQUET=1 in run.conf: Parquet copy of the FITS in <halo dir>/parquet, job_parquet.pbs)
#   optional, only when named in --stages: vpot (halo potential profiles, job_vpot.pbs; then fits again),
#   xcorr (halo-halo potential cross-correlation, job_xcorr.pbs), vprof (stacked volume-weighted potential profiles
#   around halos, job_vprof.pbs), ximult (halo correlation multipoles in real/redshift space with
#   the gravitational-redshift shift, job_ximult.pbs); production runs: postprocess.sh in the run directory
#   -> verify -> archive (checks; copy of the kept data to ARCHIVE, read-only), only when named in --stages
#   (tools/sahyadri_manager submits them itself)
# Rockstar batch: --stages rockstar --rs-snaps A-B (snapshots A..B, see job_rockstar.pbs)
#
# usage: submit_pipeline.sh <run.conf> [--stages sim,rockstar,ctrees,vahc,fits,parquet] [--after JOBID] [--hold] [--restart CHK] [--wall HH:MM:SS] [--dry-run]
#   --stages  any subset, in pipeline order (e.g. "ctrees,vahc,fits" to redo the later stages)
#   --after   make the first stage wait for an existing job (afterok), e.g. a running simulation
#   --hold    submit the first job held (qrls <id> to start the chain)
#   --restart CHK  the sim stage continues from checkpoint CHK (latest = newest $SIM_DIR/checkpoints/*.chk)
#   --wall    wall time of every stage at most HH:MM:SS (the run manager before a maintenance window; a simulation
#             is stopped before it with a STOP file, its iWallRunTime in the parameter file stays as it is)
#   --dry-run print the qsub commands only
# Prints the job ids; the chain is also written to $LOGS/chain.<run>.<date>.txt.
set -eu
[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 1; }
CONF=$(readlink -f "$1"); shift
STAGES=sim,rockstar,ctrees,vahc,fits,parquet; AFTER=; HOLD=; DRY=; RESTART=; RSSNAPS=; WALLCAP=
while [ $# -gt 0 ]; do
  case $1 in
    --stages) STAGES=$2; shift 2 ;;
    --after) AFTER=$2; shift 2 ;;
    --hold) HOLD="-h"; shift ;;
    --restart) RESTART=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --rs-snaps) RSSNAPS=$2; shift 2 ;;
    --wall) WALLCAP=$2; shift 2 ;;
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
for st in sim rockstar ctrees vahc vpot fits parquet xcorr ximult vprof verify archive; do
  case ",$STAGES," in *",$st,"*) ;; *) continue ;; esac
  [ "$st" = parquet ] && [ "${PARQUET:-0}" != 1 ] && { echo "parquet: skipped (PARQUET=${PARQUET:-0} in run.conf)"; continue; }
  case $st in
    sim)      # SIM_NODES > 1 or SIM_MPIPROCS > 1 (pegasus): pkdgrav3 with MPI ranks over the nodes (job_sim.pbs)
              # SIM_GPU=1 (pegasus GPU nodes): one GPU per node (ngpus=1), job_sim.pbs then runs build_gpu
              GPU=; [ "${SIM_GPU:-0}" = 1 ] && GPU=":ngpus=1"
              # SIM_SELECT (leap): the whole select, nodes named (SIM_NODES chunks of SIM_NCPU cores, SIM_MPIPROCS ranks)
              if [ -n "${SIM_SELECT:-}" ]; then
                R="select=$SIM_SELECT"
                [ -n "${SIM_PLACE:-}" ] && R="$R -l place=$SIM_PLACE"
              elif [ "${SIM_NODES:-1}" -gt 1 ] || [ "${SIM_MPIPROCS:-1}" -gt 1 ]; then
                R="select=${SIM_NODES:-1}:ncpus=$SIM_NCPU:mpiprocs=${SIM_MPIPROCS:-1}$GPU:mem=$SIM_MEM"
                [ -n "${SIM_PLACE:-}" ] && R="$R -l place=$SIM_PLACE"
              else
                R="select=1:ncpus=$SIM_NCPU$GPU:mem=$SIM_MEM"
              fi
              W=$SIM_WALL ;;
    rockstar) # RS_NODES > 1: Rockstar over several nodes (server here, a client per node; job_rockstar.pbs)
              R="select=${RS_NODES:-1}:ncpus=$RS_NCPU:mem=$RS_MEM"
              # RS_PLACE: placement of the nodes (default scatter:excl; scatter shares the nodes with other jobs, swarm)
              [ "${RS_NODES:-1}" -gt 1 ] && R="$R -l place=${RS_PLACE:-scatter:excl}"
              W=$RS_WALL ;;
    ctrees)   R="select=1:ncpus=$CT_NCPU:mem=$CT_MEM";     W=$CT_WALL ;;
    vahc)     R="select=1:ncpus=$PP_NCPU:mem=$PP_MEM";     W=$PP_WALL ;;
    vpot)     R="select=1:ncpus=$PP_NCPU:mem=$PP_MEM";     W=$PP_WALL ;;
    fits)     R="select=1:ncpus=$FITS_NCPU:mem=$FITS_MEM"; W=$FITS_WALL ;;
    parquet)  R="select=1:ncpus=${PQ_NCPU:-8}:mem=${PQ_MEM:-16gb}"; W=${PQ_WALL:-02:00:00} ;;
    xcorr)    R="select=1:ncpus=${XC_NCPU:-16}:mem=${XC_MEM:-16gb}"; W=${XC_WALL:-24:00:00} ;;
    ximult)   R="select=1:ncpus=${XM_NCPU:-32}:mem=${XM_MEM:-32gb}"; W=${XM_WALL:-06:00:00} ;;
    vprof)    R="select=1:ncpus=${VPROF_NCPU:-32}:mem=${VPROF_MEM:-160gb}"; W=${VPROF_WALL:-48:00:00} ;;
    verify)   R="select=1:ncpus=${VF_NCPU:-8}:mem=${VF_MEM:-32gb}"; W=${VF_WALL:-12:00:00} ;;
    archive)  R="select=1:ncpus=${AR_NCPU:-2}:mem=${AR_MEM:-8gb}"; W=${AR_WALL:-72:00:00} ;;
  esac
  if [ -n "$WALLCAP" ]; then      # HH[:MM[:SS]] -> seconds, keep the shorter
    tos() { local IFS=:; set -- $1; echo $(( 10#${1:-0} * 3600 + 10#${2:-0} * 60 + 10#${3:-0} )); }
    [ "$(tos "$WALLCAP")" -lt "$(tos "$W")" ] && W=$WALLCAP
  fi
  DEP=; [ -n "$prev" ] && DEP="-W depend=afterok:$prev"
  H=; [ -z "$prev" ] && H=$HOLD
  V="CONF=$CONF,PIPE_ENV=$HERE/pipeline_env.sh,STAGE=$st"
  [ "$st" = sim ] && [ -n "$RESTART" ] && V="$V,RESTART=$RESTART"
  RR=(); [ "$st" = sim ] && RR=(-r n)   # not rerunnable: after a node failure PBS must not start it again by itself
  N="${st}_$REAL"
  if [ "$st" = rockstar ] && [ -n "$RSSNAPS" ]; then V="$V,RS_SNAP_FIRST=${RSSNAPS%-*},RS_SNAP_LAST=${RSSNAPS#*-}"; N="rs${RSSNAPS%-*}_$REAL"; fi
  # shellcheck disable=SC2206
  RL=(-l ${R// -l / -l })      # "select=... -l place=..." -> two -l options
  PROJ=(); [ -n "${PBS_PROJECT:-}" ] && PROJ=(-P "$PBS_PROJECT")   # pegasus project queue: the project's budget
  # ${A[@]+"${A[@]}"}: an empty array under set -u is an "unbound variable" in bash < 4.4 (leap, CentOS 7: 4.2)
  cmd=("$QSUB" -N "$N" -q "${PBS_QUEUE:-workq}" ${PROJ[@]+"${PROJ[@]}"} ${RR[@]+"${RR[@]}"} "${RL[@]}" -l "walltime=$W" -j oe -o "$LOGS/pbs.$st.$RUN.out" $DEP $H
       -v "$V" "$HERE/job_$st.pbs")
  if [ -n "$DRY" ]; then echo "${cmd[*]}"; prev="<$st>"; continue; fi
  id=$("${cmd[@]}")
  echo "$st $id ($R, $W${DEP:+, $DEP})" | tee -a "$CHAIN"
  prev=$id
done
[ -n "$DRY" ] || echo "logs: $LOGS/<stage>.$RUN.<job>.log; chain: $CHAIN"
