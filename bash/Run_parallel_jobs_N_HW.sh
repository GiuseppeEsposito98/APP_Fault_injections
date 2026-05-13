#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
start_layer="$2"
stop_layer="$3"

mkdir -p ${global_PWD}/${DIR}

bash ${global_PWD}/bash/HW_Neurons_cfg_FI.sh $start_layer $stop_layer ${DIR}
