#!/bin/bash
SBATCH --job-name=UnetTrain
SBATCH --node -N2 -p biggpu
SBATCH --output=/home-mscluster/tmashinini/MSC/Code/Python/result.txt
cd /home-mscluster/tmashinini/MSC/Code/Python/
/usr/bin/python trainUNET.py

