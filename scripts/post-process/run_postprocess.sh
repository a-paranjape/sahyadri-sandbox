#!/bin/bash

if [ "$#" -ne 10 ]; then
  echo ''	
  echo "USAGE: $0 <PYTHON_EXEC> <SIM_STEM> <REAL> <NPART> <SNAP_START> <N_OUT> <PP_GRID> <LBOX> <NJOBS_PP> <NCPU_PP>"
  echo " e.g.: $0 /usr/bin/python3 sinhagad/default256 1 256 5 201 256 200.0 5 8"
  echo ''
  exit 1
fi

PYTHON_EXEC=$1
SIM_STEM=$2
REAL=$3
NPART=$4
SNAP_START=$5
N_OUT=$6
PP_GRID=$7
LBOX=$8
NJOBS_PP=$9
NCPU_PP=${10}

DOWN_SAMP=$(( NPART > 512 ? 512 : 0 )) # if particle count exceeds 512^3 then downsample to 512^3.
SNAP_END=$(( N_OUT - 1 )) # convert number of snapshots into index of last snapshot

BATCH_SIZE=$(( SNAP_END - SNAP_START + 1 ))
if [ $BATCH_SIZE -le $NJOBS_PP ]; then
    echo very few snapshots. only $BATCH_SIZE jobs needed with one cpu each
    NJOBS_PP=$BATCH_SIZE
    NCPU_PP=1
    BATCH_SIZE=1
else
    BATCH_SIZE=$(( (SNAP_END - SNAP_START + 1) / NJOBS_PP ))
    if [ $BATCH_SIZE -lt $NCPU_PP ]; then
	echo very small batches. only $BATCH_SIZE cpus needed
	NCPU_PP=$BATCH_SIZE
    fi
fi

echo batch size: $BATCH_SIZE, n_jobs: $NJOBS_PP
declare -a SJ_START
declare -a SJ_END
start=$SNAP_START
for ((i=1; i <= NJOBS_PP; i++));
do
    SJ_START[$i]=$start
    SJ_END[$i]=$((start+BATCH_SIZE-1))
    start=$((start+BATCH_SIZE))
done
SJ_END[$NJOBS_PP]=$SNAP_END # ensure last range is all-inclusive

echo job $PBS_ARRAY_INDEX:: range: "${SJ_START[$PBS_ARRAY_INDEX]} - ${SJ_END[$PBS_ARRAY_INDEX]}"

POSTPROC_EXEC=$HOME/scripts/post-process/postprocess.py # use local user version

$PYTHON_EXEC $POSTPROC_EXEC $SIM_STEM ${SJ_START[$PBS_ARRAY_INDEX]} ${SJ_END[$PBS_ARRAY_INDEX]} $REAL $PP_GRID $DOWN_SAMP $LBOX $NCPU_PP

