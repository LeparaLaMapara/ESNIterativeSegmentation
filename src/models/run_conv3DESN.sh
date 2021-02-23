#!/bin/bash
# SBATCH --job-name=UnetTrain
# SBATCH --node -N2 -p biggpu
# SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/result.txt
# cd /home-mscluster/tmashinini/MSC/Code/Python/
# /usr/bin/python trainUNET.py
for dataset in 'BSR' 'WEIZMANN' 'CIFAR_10' 'CIFAR_100'
do
    for h in 256 ;
    do 
        for lkr in 256 ;
        do
            for spt in 0 :
            do 
                for spar in 0:
                do
                    python main_conv3DESN.py 
                    --run-name=${dataset}-conv3dESN_h-${hs}_lkr-${lkr}_spt-${spt}_spar-${spar}  \
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
                    --hidden=$h \
                    --leaking-rate=$lkr\
                    --spectral-radius=$spt\
                    --sparsity=$spar

                done
            done
        done
    done
done
