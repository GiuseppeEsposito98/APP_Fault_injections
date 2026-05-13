#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
target_layer="$2"

input_args=(0 2 4)

array_size=${#input_args[@]}

mkdir -p ${global_PWD}/${DIR}

for ((i=0; i<$array_size; i++)); do
    sbatch ${global_PWD}/SLURM_scripts/Neurons_cfg_FI_ptcv.sh $target_layer ${DIR}
done