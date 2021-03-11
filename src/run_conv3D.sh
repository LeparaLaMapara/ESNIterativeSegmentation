#!/bin/bash
#SBATCH --job-name=bConv
#SBATCH --nodes=4
#SBATCH -p batch
#SBATCH --cpus-per-task=4
#SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/logs/bsr_4096_4096_0.5_conv3D.txt
#SBATCH --error=/home-mscluster/tmashinini/MSC/Code/Python/logs/bsr_4096_4096_0.5_conv3D.err

num_frames=80
num_dim=64
ch=2
lr=0.1
ep=2000
bs=32
# BSR CIFAR_100 CIFAR_10
for dataset in  'BSR';
do
    for h1 in 4096;
    do 
        for h2 in 4096;
        do
            for dp in 0.5;
            do 
                # SBATCH --job-name=${dataset}-conv3d_h1-${h1}_h2-${h2}_dp-${dp}
                # SBATCH --nodes=4
                # SBATCH --gpus-per-node=1
                # SBATCH --mem=16
                # SBATCH -p batch
                # SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/${dataset}_${h1}_${h2}_${dp}_conv3D.txt
                # cd ~/MSC/Code/Python/
                source ~/.bashrc 
                conda activate msc
                python3 /home-mscluster/tmashinini/MSC/Code/Python/main_conv3D.py --seed=16 --run-name=${dataset}-conv3d_h1-${h1}_h2-${h2}_dp-${dp} --data-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${dataset} --save-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${dataset}/results  --num-epochs=$ep  --batch-size=$bs --learning-rate=$lr --num-frames=$num_frames  --num-past-step=1 --num-future-step=1 --image-dimension=$num_dim  --threshold=0.5  --in-channels=$ch  --sample-size=$num_dim  --sample-duration=$num_frames --hidden-one=$h1 --hidden-two=$h2  --dropout-prob=$dp
            done
        done
    done
done
