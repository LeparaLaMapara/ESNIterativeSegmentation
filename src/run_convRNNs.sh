#!/bin/bash
#SBATCH --job-name=100CRNN
#SBATCH --nodes=4
#SBATCH -p batch 
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --ntasks-per-node=4      # number of tasks per node
#SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/logs/cifar100_4096_convRNN.txt
#SBATCH --error=/home-mscluster/tmashinini/MSC/Code/Python/logs/cifar100_4096_convRNN.err
num_frames=80
num_dim=64
ch=2
lr=0.1
ep=500
bs=32
# BSR WEIZMANN CIFAR_100 CIFAR_10
for dataset in 'CIFAR_100';
do
    for rnn_unit in 'RNN';
    do 
        for h in 4096;
        do
            for nlyrs in 1;
            do 
                source ~/.bashrc 
                conda activate msc
                python3 /home-mscluster/tmashinini/MSC/Code/Python/main_convRNNs.py --run-name=${dataset}-conv${rnn_unit}_h-${h}_nlyrs-$nlyrs  --data-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${dataset}  --save-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${dataset}/results  --num-epochs=$ep --batch-size=$bs --learning-rate=$lr --num-frames=$num_frames  --num-past-step=1 --num-future-step=1 --image-dimension=$num_dim --threshold=0.5 --in-channels=$ch  --sample-size=$num_dim --sample-duration=$num_frames --rnn-unit=$rnn_unit --hidden=$h  --num-layers=$nlyrs
            done
        done
    done
done
