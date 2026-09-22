#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
start_layer="$2"
stop_layer="$3"
data="$4"

input_args=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

array_size=${#input_args[@]}

mkdir -p ${global_PWD}/${DIR}

for ((i=0; i<$array_size; i++)); do
    sbatch --output=$DIR/lyr${start_layer}_${stop_layer}_stdo_%A_%a.log --error=$DIR/lyr${start_layer}_${stop_layer}_stde_%A_%a.log ${global_PWD}/SLURM_scripts/HW_Neurons_cfg_FI.sbatch $start_layer $stop_layer ${DIR} ${data}
done