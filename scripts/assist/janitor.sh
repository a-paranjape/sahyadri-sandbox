#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo ''	
  echo "USAGE: $0 <CLASS_OUT_DIR> <GADGET2_OUT_DIR> <AUTO_ROCKSTAR_DIR>"
  echo " e.g.: $0 config/transfer scratch/sims/scm1024/r1 scratch/halos/scm1024/r1"
  echo ''
  exit 1
fi

CLASS_OUT_DIR=$1
GADGET2_OUT_DIR=$2
AUTO_ROCKSTAR_DIR=$3

echo ... moving code output
mv class_run.* $CLASS_OUT_DIR/logs/.
#mv music_run.* $MUSIC_OUT_DIR/.
mv gadget_run.* $GADGET2_OUT_DIR/logs/.
mv rockstar_*.* $AUTO_ROCKSTAR_DIR/logs/.
mv clean_trees.* $AUTO_ROCKSTAR_DIR/logs/.

echo ... removing excess files
rm ics_cleanup.* dummy.* halo_cleanup.*

echo ... cleanup complete
