#!/bin/bash 

#SBATCH --job-name=data_gen
# SBATCH --gres=gpu=1
# SBATCH --gpus-per-node=1
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
# SBATCH --mem=0
# SBATCH --time=7-00:00:00
#SBATCH -p batch
#SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Matlab/Chan-Vese/result.txt
cd /home-mscluster/tmashinini/MSC/Code/Matlab/Chan-Vese
matlab -nodisplay -nosplash -nodesktop  -nojvm  -r "Readimages();exit;"



