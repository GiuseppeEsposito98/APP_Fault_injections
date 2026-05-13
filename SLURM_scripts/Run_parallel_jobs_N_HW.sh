#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
start_layer="$2"
stop_layer="$3"

input_args=(0 2 4)

array_size=${#input_args[@]}

mkdir -p ${global_PWD}/${DIR}

for ((i=0; i<$array_size; i++)); do
    sbatch ${global_PWD}/SLURM_scripts/HW_Neurons_cfg_FI.sh $start_layer $stop_layer ${DIR}
done