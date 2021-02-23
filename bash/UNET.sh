#!/bin/bash
#SBATCH --job-name=unet_train # Job name
#SBATCH -N1 -p biggpu         # -N2 number of nodes and - all the cores , -p for selecting partitio
#SBATCH --time=7-00:00:00     # How long to run for
#SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/unet_results.txt # File to which my output will be written
cd /home-mscluster/tmashinini/MSC/Code/Python/
python3 trainUNET.py

