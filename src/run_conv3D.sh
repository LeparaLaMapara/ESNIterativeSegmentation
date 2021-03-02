#!/bin/bash

#SBATCH --job-name=wconv-conv3d_h1-4096_h2-4096_dp0.5
#SBATCH --nodes=4
#SBATCH -p batch
#SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/weizmann_4069_4096_0.5_conv3D.txt
num_frames=80
num_dim=64
ch=2
for dataset in  'WEIZMANN';
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
                python3 /home-mscluster/tmashinini/MSC/Code/Python/main_conv3D.py --seed=16 --run-name=${dataset}-conv3d_h1-${h1}_h2-${h2}_dp-${dp} --data-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${dataset} --save-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${dataset}/results  --num-epochs=500  --batch-size=64 --learning-rate=0.1 --num-frames=$num_frames  --num-past-step=1 --num-future-step=1 --image-dimension=$num_dim  --threshold=0.5  --in-channels=$ch  --sample-size=$num_dim  --sample-duration=$num_frames --hidden-one=$h1 --hidden-two=$h2  --dropout-prob=$dp
            done
        done
    done
done