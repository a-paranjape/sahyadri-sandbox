#!/bin/bash

# USE WITH: bsub -m "host" -J job_name < run_rockstar.sh
# where "host" is same as used with run_rockstar_parallel.sh

if [ "$#" -ne 1 ]; then
  echo "USAGE: $0 <rockstar_config_file>"
  exit 1
fi

# Run job
/mnt/home/faculty/caseem/code/Rockstar/gfcstanford-rockstar-36ce9eea36ee/rockstar -c $1

#/mnt/home/faculty/caseem/code/Rockstar/gfcstanford-rockstar-ca79e518c38f/rockstar -c $1

