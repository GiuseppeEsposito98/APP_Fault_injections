#!/bin/bash


PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
export PYTHONPATH="$PWD"
echo ${CUDA_VISIBLE_DEVICES}


job_id=0

start_layer="$1"
stop_layer="$2"
DIR="$3"
model="$4"

Sim_dir=${global_PWD}/${DIR}/lyr${start_layer}_${stop_layer}_JOBID${job_id}_N_HW
mkdir -p ${Sim_dir}

echo ${Sim_dir}

cp ${global_PWD}/configs/Fault_descriptor.yaml ${Sim_dir}
cp ${global_PWD}/configs/evaluation_config.yaml ${Sim_dir}
sed -i "s/layers: \[.*\]/layers: \[$start_layer,$stop_layer\]/" ${Sim_dir}/Fault_descriptor.yaml
sed -i "s/trials: [0-9.]\+/trials: 5/" ${Sim_dir}/Fault_descriptor.yaml
sed -i "s/^\(\s*model_name:\s*\).*/\1'$model'/" ${Sim_dir}/evaluation_config.yaml


cd ${Sim_dir}

python ${global_PWD}/script/stm_HW_neuron_ber.py\
        --config-path ${Sim_dir}\
        --config-name evaluation_config.yaml\
        --fsim_config ${Sim_dir}/Fault_descriptor.yaml > ${global_PWD}/${DIR}/lyr${start_layer}_${stop_layer}_stdo.log