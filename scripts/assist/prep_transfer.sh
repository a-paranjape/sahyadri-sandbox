#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo ''	
  echo "USAGE: $0 <CLASS_INIT_FILE> <CLASS_TRANSFER_FILE> <PYTHON_EXEC>"
  echo " e.g.: $0 class_test.ini class_test512_pk.dat /usr/bin/python"
  echo ''
  exit 1
fi

CLASS_INIT_FILE=$1
CLASS_TRANSFER_FILE=$2
PYTHON_EXEC=$3

echo $CLASS_INIT_FILE
$HOME/code/Class/class_public/class $CLASS_INIT_FILE

mv class_output_pk.dat $CLASS_TRANSFER_FILE

$PYTHON_EXEC $HOME/scripts/assist/modify_class_output.py $CLASS_TRANSFER_FILE

echo Transfer setup complete
