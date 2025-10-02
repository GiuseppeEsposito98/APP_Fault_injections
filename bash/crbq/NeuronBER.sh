#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
target_lyr="$2"

mkdir -p ${global_PWD}/${DIR}


echo ${DIR}
export LOG_DIR=${DIR}

bash ${global_PWD}/APP_Fault_injections/bash/crbq/Neurons_cfg_FI.sh $target_lyr ${DIR} 

