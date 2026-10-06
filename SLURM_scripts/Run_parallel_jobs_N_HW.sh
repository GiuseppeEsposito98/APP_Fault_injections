#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
start_layer="$2"
stop_layer="$3"
model="$4"


mkdir -p ${global_PWD}/${DIR}

sbatch --output=$DIR/lyr${start_layer}_${stop_layer}_stdo_%A_%a.log --error=$DIR/lyr${start_layer}_${stop_layer}_stde_%A_%a.log ${global_PWD}/SLURM_scripts/HW_Neurons_cfg_FI.sbatch $start_layer $stop_layer ${DIR} ${model}