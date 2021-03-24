#!/bin/bash
#SBATCH --job-name=mpred
#SBATCH --nodes=16
#SBATCH -p batch
#SBATCH --cpus-per-task=4
#SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/logs/predictions.txt
#SBATCH --error=/home-mscluster/tmashinini/MSC/Code/Python/logs/predictions.err

num_frames=80
num_dim=64
ch=2
lr=0.1
ep=2
bs=32
h1=4096
h2=4096
dp=0.5
ds='WEIZMANN'
# source ~/.bashrc 
# conda activate msc
# python3 /home-mscluster/tmashinini/MSC/Code/Python/get_prediction_images.py 
# --run-name=predictions --data-path=/home-mscluster/tmashinini/MSC/Data/processed_data/
# --save-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${dataset}/results  --num-epochs=$ep  --batch-size=$bs 
# --learning-rate=$lr --num-frames=$num_frames  --num-past-step=1 --num-future-step=1 --image-dimension=$num_dim  
# --threshold=0.5  --in-channels=$ch  --sample-size=$num_dim  --sample-duration=$num_frames --hidden-one=$h1 --hidden-two=$h2  --dropout-prob=$dp



for dp in 0.5;
do
    for h in 4096;
    do
        for nlyrs in 1;
            do 

            source ~/.bashrc 
            conda activate msc
            python3 /home-mscluster/tmashinini/MSC/Code/Python/get_prediction_images.py --run-name=predictions --data-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${ds}  --save-path=//home-mscluster/tmashinini/MSC/Data/processed_data/${ds}/results    --batch-size=$bs  --dataset=$ds --num-frames=$num_frames  --num-past-step=1 --num-future-step=1 --image-dimension=$num_dim  --threshold=0.5  --in-channels=$ch  --sample-size=$num_dim  --sample-duration=$num_frames --hidden=$h  --dropout-prob=$dp

        done
    done
done 