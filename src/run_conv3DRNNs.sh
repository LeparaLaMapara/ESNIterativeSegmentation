#!/bin/bash
for dataset in 'CIFAR_10' 'CIFAR_100' 'BSR' 'WEIZMANN' 
do
    for rnn_unit in 'LSTM' 'GRU' 'RNN';
    do 
        for h in 128 256 512 1024;
        do
            for nlyrs in 1 2 3 :
            do 

                #SBATCH --job-name=test_${dataset}_${h}_${rnn_unit}_${nlyrs}_conv${rnn_unit}.txt
                #SBATCH --nodes=10
                #SBATCH --gpus-per-node=1
                #SBATCH --mem=16
                #SBATCH -p batch
                #SBATCH --output=~/tmashinini/MSC/Code/Python/${dataset}_${h}_${rnn_unit}_${nlyrs}_conv${rnn_unit}.txt
                "python main_conv3DRNNs.py 
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
                "
            done
        done
    done
done
