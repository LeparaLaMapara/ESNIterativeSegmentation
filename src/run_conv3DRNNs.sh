#!/bin/bash
#SBATCH --job-name=wconv3dLSTM_h-512_lkr-0.9_spy-0.2_spl-0.9
#SBATCH --nodes=4 
#SBATCH -p batch 
#SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/weizmann_512__conv3DLSTM.txt
num_frames=80
num_dim=64
ch=2
lr=0.1
ep=500
for dataset in 'WEIZMANN';
do
    for rnn_unit in 'LSTM';
    do 
        for h in 512;
        do
            for nlyrs in 1;
            do 

                # SBATCH --job-name=${dataset}-conv3d${rnn_unit}_h-${h}_nlyrs-$nlyrs
                # SBATCH --nodes=1
                # SBATCH --gpus-per-node=1
                # SBATCH --mem=16
                # SBATCH -p biggpu
                # SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/${dataset}_${h}_${rnn_unit}_${nlyrs}_conv3d${rnn_unit}.txt
                source ~/.bashrc 
                conda activate msc
                python3 /home-mscluster/tmashinini/MSC/Code/Python/main_conv3DRNNs.py --run-name=${dataset}-conv3d${rnn_unit}_h-${h}_nlyrs-$nlyrs  --data-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${dataset}  --save-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${dataset}/results  --num-epochs=$ep --batch-size=64 --learning-rate=$lr --num-frames=$num_frames  --num-past-step=1 --num-future-step=1 --image-dimension=$num_dim --threshold=0.5 --in-channels=$ch  --sample-size=$num_dim --sample-duration=$num_frames --rnn-unit=$rnn_unit --hidden=$h  --num-layers=$nlyrs
            done
        done
    done
done
