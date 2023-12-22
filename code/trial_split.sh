#!/bin/bash

#SBATCH --job-name=trial_split
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=trial_split.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abender@ucsd.edu

# Set up environment
cd /home/AD/abender/decoding_spatial_wm
source .env/bin/activate

# Call Python script
python code/trial_split.py