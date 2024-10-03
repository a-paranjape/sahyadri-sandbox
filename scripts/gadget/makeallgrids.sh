#!/bin/bash

clean=$1
clean_key="clean"
NCPU=64
#set this to zero if you also want to compile without Ngenic version
NO_NGENIC=1
#The code compile without setting double precision that positions are in single precision

for n in {7..11};
do
	MESH=$(( 2**n ))
        code_dir=$CODE_HOME/code/Gadget-4/mesh$MESH
        codeNgenic_dir=$CODE_HOME/code/Gadget-4/mesh$MESH-NGenIC

    if [[ "$clean" == "$clean_key" ]]; then
      if [[ $NO_NGENIC == 0 ]]; then
        if [ -d "$code_dir" ]; then
           echo $code_dir "exist"
           make clean DIR=$code_dir
           rm -r $code_dir 
           echo "*** removing " $code_dir
        fi
      fi

        if [ -d "$codeNgenic_dir" ]; then
           echo $codeNgenic_dir "exist"
           make clean DIR=$codeNgenic_dir
           rm -r $codeNgenic_dir 
           echo "*** removing " $codeNgenic_dir
        fi
    else
	echo "*** MESH " $MESH "***"
        #create the directory
      if [[ $NO_NGENIC == 0 ]]; then
        mkdir $code_dir
        #copy config file
        CONFIG_FILE=$code_dir/Config.sh
        echo "copying" $CODE_HOME/code/Gadget-4/gadget4/Config.sh "-->" $CONFIG_FILE
        cp $CODE_HOME/code/Gadget-4/gadget4/Config.sh $CONFIG_FILE
        # update config file
        sed -i -e "s#^PMGRID.*#PMGRID=$((2*MESH))#" "$CONFIG_FILE"
        sed -i -e "s#^DOUBLEPRECISION.*#\#DOUBLEPRECISION=1#" "$CONFIG_FILE"
        sed -i -e "s#^MAX_NUMBER_OF_RANKS_WITH_SHARED_MEMORY.*#MAX_NUMBER_OF_RANKS_WITH_SHARED_MEMORY=$NCPU#" "$CONFIG_FILE"
        sed -i -e "s#^NGENIC=.*#\#NGENIC=$MESH#" "$CONFIG_FILE"
        sed -i -e "s#^NGENIC_2LPT.*#\#NGENIC_2LPT $MESH#" "$CONFIG_FILE"
        
        #run make
        make DIR=$code_dir
      fi

        mkdir $codeNgenic_dir
        #copy config file
        CONFIG_FILE=$codeNgenic_dir/Config.sh
        echo "copying" $CODE_HOME/code/Gadget-4/gadget4/Config.sh "-->" $CONFIG_FILE
        cp $CODE_HOME/code/Gadget-4/gadget4/Config.sh $CONFIG_FILE
        # update config file
        sed -i -e "s#^PMGRID.*#PMGRID=$((2*MESH))#" "$CONFIG_FILE"
        sed -i -e "s#^DOUBLEPRECISION.*#\#DOUBLEPRECISION=1#" "$CONFIG_FILE"
        sed -i -e "s#^MAX_NUMBER_OF_RANKS_WITH_SHARED_MEMORY.*#MAX_NUMBER_OF_RANKS_WITH_SHARED_MEMORY=$NCPU#" "$CONFIG_FILE"
        sed -i -e "s#^NGENIC=.*#NGENIC=$MESH#" "$CONFIG_FILE"
        
        make DIR=$codeNgenic_dir
    fi
	#make $clean DIR=$CODE_HOME/code/Gadget-4/mesh$MESH
	#make $clean DIR=$CODE_HOME/code/Gadget-4/mesh$MESH-NGenIC
done
