#!/bin/bash
for dataset in 'CIFAR_10' 'CIFAR_100' 'BSR' 'WEIZMANN' 
do
    for h in 256 512 ;
    do 
        for lkr in 0.01 0.1 ;
        do
            for spt in 1 0.9 :
            do 
                for spar in 1 0.5 0.9:
                do
                    #SBATCH --job-name=test_${datase}_${h}_${lkr}_${spt}_${spar}_convESN.txt
                    #SBATCH --nodes=10
                    #SBATCH -p batch
                    #SBATCH --output=~/tmashinini/MSC/Code/Python/${datase}_${h}_${lkr}_${spt}_${spar}_convESN.txt
                    cd ~/tmashinini/MSC/Code/Python/
                    "
                    source activate msc; \
                    python3 main_conv3DESN.py \
                    --run-name=${dataset}-conv3dESN_h-${hs}_lkr-${lkr}_spt-${spt}_spar-${spar}  \
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
                    --hidden=$h \
                    --leaking-rate=$lkr\
                    --spectral-radius=$spt\
                    --sparsity=$spar"
                done
            done
        done
    done
done
