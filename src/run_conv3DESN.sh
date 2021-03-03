#!/bin/bash
#SBATCH --job-name=wconv3dESN_h-512_lkr-0.9_spy-0.2_spl-0.9
#SBATCH --nodes=4 
#SBATCH -p batch 
#SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/weizmann_512__conv3DESN.txt
num_frames=80
num_dim=64
ch=2
lr=0.9
ep=500
for dataset in 'WEIZMANN';
do
    for h in 512;
    do 
        for lkr in 0.9; # leaking rate
        do
            for spy in 0.2; # sparsity
            do 
                for spl in 0.9; # spectral radius
                do
                   for nlyrs in 1;
 
                        # SBATCH --job-name=${dataset}-conv3dESN_h-${h}_lkr-${lkr}_spy-${spy}_spl-${spl}
                        # SBATCH --nodes=10
                        # SBATCH --gpus-per-node=1
                        # SBATCH --mem=16
                        # SBATCH -p biggpu
                        # SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/${dataset}_${h1}_${h2}_${dp}_conv3D.txt
                        # cd ~/MSC/Code/Python/
                        source ~/.bashrc 
                        conda activate msc
                        # python main_conv3D.py --run-name=${dataset}-conv3d_h1-${h1}_h2-${h2}_dp-${dp} --data-path=/Users/thabang/Documents/msc/data/${dataset} --save-path=/Users/thabang/Documents/msc/data/${dataset}/results  --num-epochs=500  --batch-size=1 --learning-rate=0.1 --num-frames=$num_frames  --num-past-step=1 --num-future-step=1 --image-dimension=$num_dim  --threshold=0.5  --in-channels=200  --sample-size=$num_dim  --sample-duration=$num_frames --hidden-one=$h1 --hidden-two=$h2  --dropout-prob=$dp
                        python3 /home-mscluster/tmashinini/MSC/Code/Python/main_conv3DESN.py --run-name=${dataset}-conv3dESN_h-${h}_lkr-${lkr}_spy-${spy}_spl-${spl} --data-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${dataset} --save-path=/home-mscluster/tmashinini/MSC/Data/processed_data/${dataset}/results  --num-epochs=$ep  --batch-size=64 --learning-rate=$lr --num-frames=$num_frames  --num-past-step=1 --num-future-step=1 --image-dimension=$num_dim  --threshold=0.5  --in-channels=$ch  --sample-size=$num_dim  --sample-duration=$num_frames --hidden=$h  --num_layers=$nlyrs --leaking-rate=${lkr}  --sparsity=${spy} --spectral-radius=${spl}
            
                    done
                done
            done
        done
    done
done

