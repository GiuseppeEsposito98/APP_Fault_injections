#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
data="$2"

input_args=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
#   

array_size=${#input_args[@]}

mkdir -p ${global_PWD}/${DIR}

for ((i=0; i<$array_size; i++)); do
    sbatch --output=$DIR/lyr${input_args[$((i))]}_stdo_%A_%a.log --error=$DIR/lyr${input_args[$((i))]}_stde_%A_%a.log ${global_PWD}/SLURM_scripts/Neurons_cfg_FI_ptcv.sbatch ${input_args[$((i))]} ${DIR} ${data}
done

# --error=$DIR/lyr${input_args[$((i))]}_stde_%A_%a.log
