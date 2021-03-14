#!/bin/bash
#SBATCH --job-name=bConv
#SBATCH --nodes=1
#SBATCH -p batch
#SBATCH --cpus-per-task=4
#SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/logs/predictions.txt
#SBATCH --error=/home-mscluster/tmashinini/MSC/Code/Python/logs/predictions.err

num_frames=80
num_dim=64
ch=2
lr=0.1
ep=2000
bs=32
dataset='BSR'
h1=4096
h2=4096
dp=0.5
source ~/.bashrc 
conda activate msc
python3 /home-mscluster/tmashinini/MSC/Code/Python/get_prediction_images.py --data-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${dataset} --save-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${dataset}/results  --num-epochs=$ep  --batch-size=$bs --learning-rate=$lr --num-frames=$num_frames  --num-past-step=1 --num-future-step=1 --image-dimension=$num_dim  --threshold=0.5  --in-channels=$ch  --sample-size=$num_dim  --sample-duration=$num_frames --hidden-one=$h1 --hidden-two=$h2  --dropout-prob=$dp