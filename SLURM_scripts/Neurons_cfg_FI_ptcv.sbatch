#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=Neurons
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=40000
#SBATCH --mail-user=giuseppe.esposito@polito.it

# 1 Activate the virtual environment
source ~/miniconda3/bin/activate
conda deactivate

cd  ~/APP_Fault_injections

conda activate APP
# module load nvidia/cudasdk/11.6

PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
export PYTHONPATH="$PWD"
echo ${CUDA_VISIBLE_DEVICES}

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.backends.cudnn.version()); print(torch.cuda.is_available())"

job_id=0

target_lyr="$1"
DIR="$2"
data="$3"

Sim_dir=${global_PWD}/${DIR}/lyr${target_lyr}_JOBID${job_id}_N
mkdir -p ${Sim_dir}

cp ${global_PWD}/configs/Fault_descriptor.yaml ${Sim_dir}
sed -i "s/layers: \[.*\]/layers: \[$target_lyr\]/" ${Sim_dir}/Fault_descriptor.yaml
sed -i "s/trials: [0-9.]\+/trials: 5/" ${Sim_dir}/Fault_descriptor.yaml

cd ${Sim_dir}

python ${global_PWD}/script/resnet20_neuron_ber.py\
        --fsim_config ${Sim_dir}/Fault_descriptor.yaml --data ${data} > ${global_PWD}/${DIR}/lyr${target_layer}_stdo.log # 2> ${global_PWD}/${DIR}/lyr${target_layer}_stde.log