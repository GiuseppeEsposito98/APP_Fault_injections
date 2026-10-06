#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
target_layer="$2"
model="$3"

mkdir -p ${global_PWD}/${DIR}

bash ${global_PWD}/bash/stm/Neurons_cfg_FI_ptcv.sh $target_layer ${DIR} ${model}
