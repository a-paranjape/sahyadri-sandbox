#!/bin/bash

if [ "$#" -ne 6 ]; then
  echo ''	
  echo "USAGE: $0 <GADGET2_CONFIG_FILE> <NPART> <NCPU_TOT> <NGENIC> <RESTART> <CODE_HOME>"
  echo " e.g.: $0 $HOME/config/sims/run.param 1024 256 0 0 /mnt/home/faculty/caseem"
  echo ''
  exit 1
fi

GADGET2_CONFIG_FILE=$1
NPART=$2
NCPU_TOT=$3
NMESH=$(( NPART * 2 ))
NGENIC=$4
RESTART=$5
CODE_HOME=$6

# Run MPI program
#mpirun -np $NCPU_TOT -envall $HOME/code/Gadget-2/Gadget-2.0.7-mesh$NMESH/Gadget2/Gadget2 $GADGET2_CONFIG_FILE $RESTART
if [ $NGENIC == 1 ]; then
    if [ $RESTART == 0 ]; then
        mpirun -np $NCPU_TOT -machinefile $PBS_NODEFILE $CODE_HOME/code/Gadget-4/mesh$NMESH-NGenIC/Gadget4 $GADGET2_CONFIG_FILE 
        # mpiexec -np $NCPU_TOT $CODE_HOME/code/Gadget-4/mesh$NMESH-NGenIC/Gadget4 $GADGET2_CONFIG_FILE 
    else
        mpirun -np $NCPU_TOT -machinefile $PBS_NODEFILE $CODE_HOME/code/Gadget-4/mesh$NMESH-NGenIC/Gadget4 $GADGET2_CONFIG_FILE $RESTART
        #mpiexec -np $NCPU_TOT $CODE_HOME/code/Gadget-4/mesh$NMESH-NGenIC/Gadget4 $GADGET2_CONFIG_FILE $RESTART
    fi
else
    if [ $RESTART == 0 ]; then
        mpirun -np $NCPU_TOT -machinefile $PBS_NODEFILE $CODE_HOME/code/Gadget-4/mesh$NMESH/Gadget4 $GADGET2_CONFIG_FILE
        #mpiexec -np $NCPU_TOT $CODE_HOME/code/Gadget-4/mesh$NMESH/Gadget4 $GADGET2_CONFIG_FILE
    else
        mpirun -np $NCPU_TOT -machinefile $PBS_NODEFILE $CODE_HOME/code/Gadget-4/mesh$NMESH/Gadget4 $GADGET2_CONFIG_FILE $RESTART
        #mpiexec -np $NCPU_TOT $CODE_HOME/code/Gadget-4/mesh$NMESH/Gadget4 $GADGET2_CONFIG_FILE $RESTART
    fi
fi
