#!/bin/bash
for dataset in 'BSR';
do
    for rnn_unit in 'LSTM';
    do 
        for h in 128;
        do
            for nlyrs in 1;
            do 

                # SBATCH --job-name=${dataset}-conv3d${rnn_unit}_h-${h}_nlyrs-$nlyrs
                # SBATCH --nodes=1
                # SBATCH --gpus-per-node=1
                # SBATCH --mem=16
                # SBATCH -p biggpu
                # SBATCH --output=~/MSC/Code/Python/${dataset}_${h}_${rnn_unit}_${nlyrs}_conv3d${rnn_unit}.txt
                source ~/.bashrc 
                conda activate msc
                python3 /home-mscluster/tmashinini/MSC/Code/Python/main_conv3DRNNs.py --run-name=${dataset}-conv3d${rnn_unit}_h-${h}_nlyrs-$nlyrs  --data-path=~/MSC/Data/processed_data/${dataset}  --save-path=~/MSC/Data/processed_data/${dataset}/results  --num-epochs=500 --batch-size=64 --learning-rate=0.1 --num-frames=95 --num-past-step=1 --num-future-step=1 --image-dimension=96 --threshold=0.5 --in-channels=1  --sample-size=128 --sample-duration=16 --rnn-unit=$rnn_unit --hidden=$h  --num-layers=$nlyrs
            done
        done
    done
done
