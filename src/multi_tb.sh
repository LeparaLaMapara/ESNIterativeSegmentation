#!/bin/bash

# multitb() {
#     #SBATCH --ntasks=1
#     #SBATCH -t 04:00:00               # max runtime is 4 hours
#     #SBATCH -J  tensorboard_server    # name
#     #SBATCH  --partition=batch 
#     #SBATCH -o /home-mscluster/tmashinini/MSC/Code/Python/tb-%J.out #TODO: Where to save your output
#     source /home-mscluster/tmashinini/.bash_profile
#     logdir=
#     if [ $# -eq 0 ]; then
#         printf >&2 'fatal: provide at least one logdir\n'
#     fi
#     for arg; do
#         logdir="${logdir}${logdir:+,}${arg}"
#     done
#     (set -x; tensorboard --port 8776 --logdir_spec="${logdir}")
# }

# #echo "$@"
# multitb "$@"




#!/bin/sh

#SBATCH --ntasks=1
#SBATCH -t 04:00:00               # max runtime is 4 hours
#SBATCH -J  tensorboard_server    # name
#SBATCH  --partition=batch 
#SBATCH -o /home-mscluster/tmashinini/MSC/Code/Python/tb-%J.out #TODO: Where to save your output

# To run as an array job, use the following command:
# sbatch --partition=beards --array=0-0 tensorboardHam.sh
# squeue --user thpaul

source /home-mscluster/tmashinini/.bash_profile #TODO: Your profile
source /home-mscluster/tmashinini/.bashrc #TODO: Your profile
conda activate msc
MODEL_DIR=/home-mscluster/tmashinini/MSC/Data/processed_data/BSR/results/BSR-conv3d_h1-256_h2-512_dp-0.5/tensorboard_logs/ #TODO: Your TF model directory

let ipnport=($UID-6025)%65274
echo ipnport=$ipnport

ipnip=$(hostname -i)
echo ipnip=$ipnip
echo $MODEL_DIR

set -x;tensorboard --logdir="${MODEL_DIR}" --port=$ipnport