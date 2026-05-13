#!/bin/bash


PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
export PYTHONPATH="$PWD"
echo ${CUDA_VISIBLE_DEVICES}


job_id=0

target_lyr="$1"
DIR="$2"

Sim_dir=${global_PWD}/${DIR}/lyr${target_lyr}_JOBID${job_id}_N
mkdir -p ${Sim_dir}

cp ${global_PWD}/configs/Fault_descriptor.yaml ${Sim_dir}
sed -i "s/layers: \[.*\]/layers: \[$target_lyr\]/" ${Sim_dir}/Fault_descriptor.yaml
sed -i "s/trials: [0-9.]\+/trials: 5/" ${Sim_dir}/Fault_descriptor.yaml

cd ${Sim_dir}

python ${global_PWD}/script/resnet20_neuron_ber.py\
        --fsim_config ${Sim_dir}/Fault_descriptor.yaml > ${global_PWD}/${DIR}/lyr${target_layer}_stdo.log # 2> ${global_PWD}/${DIR}/lyr${target_layer}_stde.log

