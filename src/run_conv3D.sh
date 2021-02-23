#!/bin/bash
# SBATCH --job-name=UnetTrain
# SBATCH --node -N2 -p biggpu
# SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/result.txt
# cd /home-mscluster/tmashinini/MSC/Code/Python/
# /usr/bin/python trainUNET.py
dataset=BSR
for h1 in 256 ;
do 
    for h2 in 256 ;
    do
        for dp in 0 :
        do 

            python main_conv3D.py --run-name=conv3d_h1-${hs}_h2-${h2}_dropoutprob-${dp}  \
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
            --hidden-one=$h2 \
            --hidden-two=$h2  \
            # --dropout-prob=$dp
       done
    done
done