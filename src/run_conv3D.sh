#!/bin/bash
for dataset in 'CIFAR_10' 'CIFAR_100' 'BSR' 'WEIZMANN' ;
do
    for h1 in 256 512 1024 ;
    do 
        for h2 in 256 512 1024;
        do
            for dp in 0.5;
            do 
                #SBATCH --job-name=test_${dataset}_${h1}_${h2}_${dp}_conv3D.txt
                #SBATCH --nodes=10
                #SBATCH --gpus-per-node=1
                #SBATCH --mem=16
                #SBATCH -p batch
                #SBATCH --output=~/MSC/Code/Python/${dataset}_${h1}_${h2}_${dp}_conv3D.txt
                cd ~/MSC/Code/Python/
                source ~/.bashrc 
                conda activate msc
                python main_conv3D.py 
                --run-name=${dataset}-conv3d_h1-${h1}_h2-${h2}_dp-${dp}  \
                --data-path=~/MSC/Data/processed_data/${dataset} \
                --save-path=~/MSC/Data/processed_data/${dataset}/results \
                --num-epochs=500 \
                --batch-size=64 \
                --learning-rate=0.1 \
                --num-frames=95  \
                --num-past-step=1  \
                --num-future-step=1  \
                --image-dimension=96  \
                --threshold=0.5 \
                --in-channels=1  \
                --sample-size=128  \
                --sample-duration=16\
                --hidden-one=$h1 \
                --hidden-two=$h2  \
                --dropout-prob=$dp
                
            done
        done
    done
done