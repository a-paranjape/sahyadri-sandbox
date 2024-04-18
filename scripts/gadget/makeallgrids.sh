#!/bin/bash

clean=$1

for n in {7..11};
do
	MESH=$(( 2**n ))
	#echo $MESH
	make $clean DIR=/mnt/home/faculty/caseem/code/Gadget-4/mesh$MESH
	make $clean DIR=/mnt/home/faculty/caseem/code/Gadget-4/mesh$MESH-NGenIC
done
