#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "USAGE: $0 <SUB_CONFIG_FILE>"
  echo " e.g.: $0 sub_scmWMAP7.conf"
  exit 1
fi

# general setup
SCF=$HOME/config/submit/$1
SIM_NAME=`awk '$1=="SIM_NAME" {print $3}' $SCF`
SIM_NPART=`awk '$1=="SIM_NPART" {print $3}' $SCF`
SIM_REAL=`awk '$1=="SIM_REAL" {print $3}' $SCF`
SIM_SEED=`awk '$1=="SIM_SEED" {print $3}' $SCF`

SIM_STUB=$SIM_NAME$SIM_NPART

# Simulation
LBOX=`awk '$1=="LBOX" {print $3}' $SCF`
A_START=`awk '$1=="A_START" {print $3}' $SCF`
Z_START=`awk '$1=="Z_START" {print $3}' $SCF`
A_FIN=`awk '$1=="A_FIN" {print $3}' $SCF`
FORCE_RES=`awk '$1=="FORCE_RES" {print $3}' $SCF`
OUTPUTS_TXT=`awk '$1=="OUTPUTS_TXT" {print $3}' $SCF`
TRANSFER_CODE=`awk '$1=="TRANSFER_CODE" {print $3}' $SCF`
NGENIC=`awk '$1=="NGENIC" {print $3}' $SCF`
RUN_SIM=`awk '$1=="RUN_SIM" {print $3}' $SCF`
RUN_CLASS=`awk '$1=="RUN_CLASS" {print $3}' $SCF`
RESTART=`awk '$1=="RESTART" {print $3}' $SCF`

CLASS_OUT_DIR=$HOME/config/transfer # always needed
CLASS_TRANSFER_ROOT=$CLASS_OUT_DIR/class_$SIM_STUB
CLASS_TRANSFER=$CLASS_TRANSFER_ROOT\_pk.txt
CLASS_TRANSFER_RAW=$CLASS_TRANSFER_ROOT\_pk.dat

# Cosmology
OMEGA_M=`awk '$1=="OMEGA_M" {print $3}' $SCF`
OMEGA_L=`awk '$1=="OMEGA_L" {print $3}' $SCF`
OMEGA_K=`awk '$1=="OMEGA_K" {print $3}' $SCF`
OMEGA_B=`awk '$1=="OMEGA_B" {print $3}' $SCF`
HUBBLE=`awk '$1=="HUBBLE" {print $3}' $SCF`
H0=`awk '$1=="H0" {print $3}' $SCF`
SIGMA8=`awk '$1=="SIGMA8" {print $3}' $SCF`
NS=`awk '$1=="NS" {print $3}' $SCF`

# Run Rockstar / consistent trees?
HALOS=`awk '$1=="HALOS" {print $3}' $SCF`
TREES=`awk '$1=="TREES" {print $3}' $SCF`

NHRS=192
NMIN=00
NCPU=32
NNODE=1
LGNPART=6
NFILE=1
NWRITER=16
case $SIM_NPART in
  128)
    NHRS=05; NMIN=00; NCPU=16; LGNPART=7; NWRITER=8
    ;;
  256)
    NHRS=12; NNODE=2; LGNPART=8
    ;;
  512)
    NHRS=24; NNODE=4; LGNPART=9
    ;;
  1024)
    NNODE=8; LGNPART=10; NFILE=8
    ;;
esac
NCPU_TOT=$(( NCPU * NNODE ))

# setup dummy job and janitor
DUMMY_EXEC=$HOME/scripts/assist/dummy.sh
JANITOR=$HOME/scripts/assist/janitor.sh

# setup CLASS run and Python
PREP_TRANSFER=$HOME/scripts/assist/prep\_transfer.sh
PYTHON_EXEC=/mnt/csoft/tools/anaconda3/bin/python
CLASS_TEMPLATE=$CLASS_OUT_DIR/class_template_sig8.ini # adjust later for non-standard CDM
CLASS_CONFIG_FILE=$CLASS_OUT_DIR/class_$SIM_STUB.ini

# setup GADGET
GADGET_OUT_DIR=$SCRATCH/sims/$SIM_STUB/r$SIM_REAL
GADGET_CONFIG_FILE=$HOME/config/sims/run.param.$SIM_STUB.r$SIM_REAL
GADGET_EXEC=$HOME/scripts/gadget/run\_gadget.sh 
GADGET_TEMPLATE=$HOME/config/sims/run.param.template

# setup NGenIC 
if [ $RESTART == 6 ]; then
    if [ $RUN_SIM == 0 ]; then
	echo "RESTART = 6 means simulation must start. Setting RUN_SIM = 1."
	RUN_SIM=1
    fi
    if [ $NGENIC == 0 ]; then
	echo "RESTART = 6 means NGenIC output needed. Setting NGENIC = 1."
	NGENIC=1
    fi
    if [ $HALOS == 1 ]; then
	echo "RESTART = 6 means simulation will stop after IC generation. Setting HALOS = 0."
	HALOS=0
    fi
    if [ $TREES == 1 ]; then
	echo "RESTART = 6 means halos will not be found. Setting TREES = 0."
	TREES=0
    fi
else
    NGENIC_OUT_FILE=$GADGET_OUT_DIR/snapshot_ics_000.hdf5
fi


# setup Rockstar
ROCKSTAR_STUB=$SIM_STUB
ROCKSTAR_CONFIG_FILE=$HOME/config/halos/rockstar\_$ROCKSTAR_STUB\_r$SIM_REAL.cfg
AUTO_ROCKSTAR_DIR=$SCRATCH/halos/$ROCKSTAR_STUB/r$SIM_REAL
ROCKSTAR_EXEC=$HOME/scripts/rockstar/run\_rockstar.sh
ROCKSTAR_TEMPLATE=$HOME/config/halos/rockstar\_template.cfg

# setup analysis script
GADGET_LITE_EXEC=$HOME/scripts/util/gadget-lite.sh

# setup perl
PERL_EXEC=/usr/bin/perl

# set names for jobs
CLASS_RUN=class_run
GADGET_RUN=gadget\_run
ROCKSTAR_SERV=rockstar\_serv
ROCKSTAR_PROC=rockstar\_proc

#################
echo 'checking that output directories exist'
if [ ! -d $GADGET_OUT_DIR ]; then
  echo "making directory: $GADGET_OUT_DIR"
  mkdir -p $GADGET_OUT_DIR
  mkdir $GADGET_OUT_DIR/logs
fi

echo "copying outputs file to: $SCRATCH/sims/$SIM_STUB"
cp -t $SCRATCH/sims/$SIM_STUB $OUTPUTS_TXT 

if [ ! -d $AUTO_ROCKSTAR_DIR ]; then
  echo "making directory: $AUTO_ROCKSTAR_DIR"
  mkdir -p $AUTO_ROCKSTAR_DIR
  mkdir $AUTO_ROCKSTAR_DIR/logs
fi

N_OUT=`awk 'END {print FNR}' $OUTPUTS_TXT`
N_OUT=$(( N_OUT + 1 ))

echo 'writing config files'
echo '... class'
cp $CLASS_TEMPLATE $CLASS_CONFIG_FILE
sed -i -e "s#^H0.*#H0\t= $H0#" "$CLASS_CONFIG_FILE"
sed -i -e "s#^Omega_b.*#Omega_b = $OMEGA_B#" "$CLASS_CONFIG_FILE"
sed -i -e "s#^Omega_m.*#Omega_m = $OMEGA_M#" "$CLASS_CONFIG_FILE"
sed -i -e "s#^Omega_k.*#Omega_k = $OMEGA_K#" "$CLASS_CONFIG_FILE"
sed -i -e "s#^sigma8.*#sigma8 = $SIGMA8#" "$CLASS_CONFIG_FILE"
sed -i -e "s#^n_s.*#n_s = $NS#" "$CLASS_CONFIG_FILE"
sed -i -e "s#^z_pk.*#z_pk = $Z_START#" "$CLASS_CONFIG_FILE"


echo '... gadget'
cp $GADGET_TEMPLATE $GADGET_CONFIG_FILE
sed -i -e "s#OutputDir.*#OutputDir\t $GADGET_OUT_DIR#" "$GADGET_CONFIG_FILE"
sed -i -e "s#OutputListFilename.*#OutputListFilename\t $OUTPUTS_TXT#" "$GADGET_CONFIG_FILE"

sed -i -e "s/^TimeBegin.*/TimeBegin\t\t $A_START/" "$GADGET_CONFIG_FILE"
sed -i -e "s/^TimeMax.*/TimeMax\t\t $A_FIN/" "$GADGET_CONFIG_FILE"

sed -i -e "s/^Omega0.*/Omega0\t\t $OMEGA_M/" "$GADGET_CONFIG_FILE"
sed -i -e "s/^OmegaLambda.*/OmegaLambda\t $OMEGA_L/" "$GADGET_CONFIG_FILE"
sed -i -e "s/^OmegaBaryon.*/OmegaBaryon\t $OMEGA_B/" "$GADGET_CONFIG_FILE"
sed -i -e "s/^HubbleParam.*/HubbleParam\t $HUBBLE/" "$GADGET_CONFIG_FILE"
sed -i -e "s/^BoxSize.*/BoxSize\t\t $LBOX/" "$GADGET_CONFIG_FILE"

sed -i -e "s/^NumFilesPerSnapshot.*/NumFilesPerSnapshot\t\t $NFILE/" "$GADGET_CONFIG_FILE"
sed -i -e "s/^MaxFilesWithConcurrentIO.*/MaxFilesWithConcurrentIO\t $NFILE/" "$GADGET_CONFIG_FILE"

sed -i -e "s/^SofteningComovingClass0 .*/SofteningComovingClass0\t $FORCE_RES/" "$GADGET_CONFIG_FILE"
sed -i -e "s/^SofteningMaxPhysClass0.*/SofteningMaxPhysClass0\t $FORCE_RES/" "$GADGET_CONFIG_FILE"
if [ $NGENIC == 1 ]; then
    PK_TYPE=2
    SPEC_INDEX=1.0
    R_PS=0
    if [ $TRANSFER_CODE == eisenstein ]; then
	PK_TYPE=1
	SPEC_INDEX=$NS
	R_PS=1
    fi
    sed -i -e "s/^NSample .*/NSample\t $SIM_NPART/" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^GridSize .*/GridSize\t $SIM_NPART/" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^Seed .*/Seed\t $SIM_SEED/" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^SphereMode .*/SphereMode\t 0/" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^PowerSpectrumType .*/PowerSpectrumType\t $PK_TYPE /" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^ReNormalizeInputSpectrum .*/ReNormalizeInputSpectrum\t $R_PS /" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^PrimordialIndex .*/PrimordialIndex\t $SPEC_INDEX /" "$GADGET_CONFIG_FILE"
    sed -i -e "s#^PowerSpectrumFile.*#PowerSpectrumFile\t $CLASS_TRANSFER#" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^Sigma8 .*/Sigma8\t $SIGMA8 /" "$GADGET_CONFIG_FILE" # redundant for class transfer, needed for eisenstein
else
    sed -i -e "s#InitCondFile.*#InitCondFile\t $NGENIC_OUT_FILE#" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^NSample .*/\t /" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^GridSize .*/\t /" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^Seed .*/\t /" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^SphereMode .*/\t /" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^PowerSpectrumType .*/\t /" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^ReNormalizeInputSpectrum .*/\t /" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^PrimordialIndex .*/\t /" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^ShapeGamma .*/\t /" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^Sigma8 .*/\t /" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^PowerSpectrumFile .*/\t /" "$GADGET_CONFIG_FILE"
    sed -i -e "s/^InputSpectrum_UnitLength_in_cm .*/\t /" "$GADGET_CONFIG_FILE"
fi

echo '... rockstar'
cp $ROCKSTAR_TEMPLATE $ROCKSTAR_CONFIG_FILE
sed -i -e "s#^INBASE.*#INBASE = \"$GADGET_OUT_DIR\"#" "$ROCKSTAR_CONFIG_FILE"
sed -i -e "s#^OUTBASE.*#OUTBASE = \"$AUTO_ROCKSTAR_DIR\"#" "$ROCKSTAR_CONFIG_FILE"
sed -i -e "s#^FORCE\_RES.*#FORCE\_RES = $FORCE_RES#" "$ROCKSTAR_CONFIG_FILE"
sed -i -e "s#^NUM\_SNAPS.*#NUM\_SNAPS = $N_OUT#" "$ROCKSTAR_CONFIG_FILE"
sed -i -e "s#^NUM\_BLOCKS.*#NUM\_BLOCKS = $NFILE#" "$ROCKSTAR_CONFIG_FILE"
sed -i -e "s#^NUM\_WRITERS.*#NUM\_WRITERS = $NWRITER#" "$ROCKSTAR_CONFIG_FILE"
sed -i -e "s#^FORK\_PROCESSORS\_PER\_MACHINE.*#FORK\_PROCESSORS\_PER\_MACHINE = $NWRITER#" "$ROCKSTAR_CONFIG_FILE"
if [ "$NFILE" -ne 1 ]; then
    sed -i -e "s#^FILENAME.*#FILENAME = \"snapshot_<snap>/snapshot_<snap>.<block>.hdf5\"#" "$ROCKSTAR_CONFIG_FILE"
fi

echo "... all good!     proceeding with submission"
#################

# run CLASS and setup transfer function
if [ $RUN_CLASS == 1 ]; then
    echo "submitting class job"
    CLASS_JOB=`qsub -V -N $CLASS_RUN -k oe -l walltime=01:00:00 -l select=1:ncpus=16 -- $PREP_TRANSFER $CLASS_CONFIG_FILE $CLASS_TRANSFER_RAW $PYTHON_EXEC`
else
    echo "transfer function not requested"
    cd $HOME
    CLASS_JOB=`qsub -N dummy -k oe  -- $DUMMY_EXEC`
fi

# run gadget
if [ $RUN_SIM == 1 ]; then
    echo "submitting gadget job"
    cd $GADGET_OUT_DIR
    GADGET_JOB=`qsub -V -N $GADGET_RUN -k oe -W depend=afterok:$CLASS_JOB -l walltime=$NHRS:$NMIN:00 -l select=$NNODE:ncpus=$NCPU:mpiprocs=$NCPU -l place=pack:exclhost -- $GADGET_EXEC $GADGET_CONFIG_FILE $SIM_NPART $NCPU_TOT $NGENIC $RESTART`
else
    echo "simulation not requested"
    cd $HOME
    GADGET_JOB=`qsub -N dummy -k oe -W depend=afterok:$CLASS_JOB  -- $DUMMY_EXEC`
fi

if [ $HALOS == 1 ]; then
    echo "submitting rockstar job"
    # run Rockstar
    cd $AUTO_ROCKSTAR_DIR
    ROCKSTAR_SERV_JOB=`qsub -V -N $ROCKSTAR_SERV -k oe -W depend=afterok:$GADGET_JOB -l walltime=05:00:00 -- $ROCKSTAR_EXEC $ROCKSTAR_CONFIG_FILE`
    ROCKSTAR_PROC_JOB=`qsub -V -N $ROCKSTAR_PROC -k oe -W depend=after:$ROCKSTAR_SERV_JOB -l walltime=05:00:00 -l select=ncpus=$NWRITER -- $ROCKSTAR_EXEC $AUTO_ROCKSTAR_DIR/auto-rockstar.cfg`

    # run ConsistentTrees if requested
    if [ $TREES ==  1 ]; then
      CONSISTENT_TREES_GENCFG=$HOME/code/Rockstar/gfcstanford-rockstar-36ce9eea36ee/scripts/gen\_merger\_cfg.pl
      GEN_CFG=ctrees\_cfg
      GEN_TREES=ctrees\_trees
      GEN_CAT=ctrees\_cat
      cd $AUTO_ROCKSTAR_DIR
      echo "submitting consistent trees job"
      GEN_CFG_JOB=`qsub -V -N $GEN_CFG -k oe -W depend=afterok:$ROCKSTAR_SERV_JOB -l walltime=00:10:00 -- $PERL_EXEC $CONSISTENT_TREES_GENCFG $ROCKSTAR_CONFIG_FILE`
      CONSISTENT_TREES_DIR=$HOME/code/ConsistentTrees/consistent-trees
      CONSISTENT_TREES_CONFIG_FILE=$AUTO_ROCKSTAR_DIR/outputs/merger\_tree.cfg
      TREES_EXEC=$CONSISTENT_TREES_DIR/do\_merger\_tree.pl
      CATALOG_EXEC=$CONSISTENT_TREES_DIR/halo\_trees\_to\_catalog.pl
      cd $CONSISTENT_TREES_DIR
      GEN_TREES_JOB=`qsub -V -N $GEN_TREES -k oe -W depend=afterok:$GEN_CFG_JOB -l walltime=05:00:00 -l select=ncpus=$NWRITER -- $PERL_EXEC $TREES_EXEC $CONSISTENT_TREES_CONFIG_FILE $CONSISTENT_TREES_DIR`
      GEN_CAT_JOB=`qsub -V -N $GEN_CAT -k oe -W depend=afterok:$GEN_TREES_JOB -l walltime=00:30:00 -- $PERL_EXEC $CATALOG_EXEC $CONSISTENT_TREES_CONFIG_FILE $CONSISTENT_TREES_DIR`

      CLEAN_TREE_EXEC=$HOME/scripts/assist/clean\_trees.pl
      CLEAN_TREE_JOB=`qsub -V -N clean\_trees -k oe -W depend=afterok:$GEN_CAT_JOB -l walltime=00:10:00 -- $PERL_EXEC $CLEAN_TREE_EXEC $AUTO_ROCKSTAR_DIR`
      ROCKSTAR_SERV_JOB=$CLEAN_TREE_JOB
    fi
    # #################

    cd $HOME
    clean=halo_cleanup.sh
    str=$'#!/usr/bin/bash\n\n'
    str="${str}sleep 2; rm -rf $AUTO_ROCKSTAR_DIR/profiling/ $AUTO_ROCKSTAR_DIR/*.ascii $AUTO_ROCKSTAR_DIR/*.bin; "
    if [ $TREES == 1 ]; then
        str="${str}mv ctrees_cfg*.* $AUTO_ROCKSTAR_DIR/logs/.; mv ctrees_trees*.* $AUTO_ROCKSTAR_DIR/logs/.; mv ctrees_cat*.* $AUTO_ROCKSTAR_DIR/logs/."
    fi
    echo "$str" > $clean
    chmod +x $clean
    HALO_CLEANUP_JOB=`qsub -N halo_cleanup -W depend=afterok:$ROCKSTAR_SERV_JOB -- $clean`

    cd $HOME
    qsub -N janitor -k oe -W depend=afterok:$HALO_CLEANUP_JOB -l walltime=00:10:00 -- $JANITOR $CLASS_OUT_DIR $GADGET_OUT_DIR $AUTO_ROCKSTAR_DIR
else
    echo halos not requested
    cd $HOME
    qsub -N janitor -k oe -W depend=afterok:$GADGET_JOB -l walltime=00:10:00 -- $JANITOR $CLASS_OUT_DIR $GADGET_OUT_DIR $AUTO_ROCKSTAR_DIR
fi