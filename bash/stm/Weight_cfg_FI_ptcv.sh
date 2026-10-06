#!/bin/bash

PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
export PYTHONPATH="$PWD"
echo ${CUDA_VISIBLE_DEVICES}


job_id=0

target_layer="$1"
DIR="$2"
model="$3"

# config_dir="configs_imagenet_pt/STD"

Sim_dir=${global_PWD}/${DIR}/lyr${target_layer}_JOBID${job_id}_W
mkdir -p ${Sim_dir}

cp ${global_PWD}/configs/Fault_descriptor.yaml ${Sim_dir}
cp ${global_PWD}/configs/evaluation_config.yaml ${Sim_dir}
sed -i "s/layer: \[.*\]/layer: \[$target_layer\]/" ${Sim_dir}/Fault_descriptor.yaml
sed -i "s/^\(\s*model_name:\s*\).*/\1'$model'/" ${Sim_dir}/evaluation_config.yaml

cd ${Sim_dir}

python ${global_PWD}/script/stm_sbfm.py\
        --config-path ${Sim_dir}\
        --config-name evaluation_config.yaml\
        --fsim_config ${Sim_dir}/Fault_descriptor.yaml > ${global_PWD}/${DIR}/lyr${target_layer}_stdo.log