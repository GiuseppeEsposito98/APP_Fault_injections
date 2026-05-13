#!/bin/bash

PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
export PYTHONPATH="$PWD"
echo ${CUDA_VISIBLE_DEVICES}


job_id=0

target_layer="$1"
DIR="$2"

Sim_dir=${global_PWD}/${DIR}/lyr${target_layer}_JOBID${job_id}_W
mkdir -p ${Sim_dir}

cp ${global_PWD}/configs/Fault_descriptor.yaml ${Sim_dir}
sed -i "s/layer: \[.*\]/layer: \[$target_layer\]/" ${Sim_dir}/Fault_descriptor.yaml

cd ${Sim_dir}

python ${global_PWD}/script/resnet20_sbfm.py\
        --fsim_config ${Sim_dir}/Fault_descriptor.yaml > ${global_PWD}/${DIR}/lyr${target_layer}_stdo.log # 2> ${global_PWD}/${DIR}/lyr${target_layer}_stde.log

