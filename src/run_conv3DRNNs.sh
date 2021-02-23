#!/bin/bash
# SBATCH --job-name=UnetTrain
# SBATCH --node -N2 -p biggpu
# SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/result.txt
# cd /home-mscluster/tmashinini/MSC/Code/Python/
# /usr/bin/python trainUNET.py
for dataset in 'BSR' 'WEIZMANN' 'CIFAR_10' 'CIFAR_100'
do
    for rnn_unit in 'LSTM' 'GRU' 'RNN';
    do 
        for h in 128 256 512 1024;
        do
            for nlyrs in 1 2 3 :
            do 
                python main_conv3DRNNs.py 
                --run-name=${dataset}-conv3d${rnn_unit}_h-${h}_nlyrs-$nlyrs  \
                --data-path=F:/MSC/Data/processed_data/${dataset} \
                --save-path=F:/MSC/Data/processed_data/${dataset}/results \
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
                ----rnn-unit=$rnn_unit \
                --hidden=$h  \
                --num-layers=$nlyrs
            done
        done
    done
done
