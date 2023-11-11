#!/bin/bash

#SBATCH --job-name=count_psds
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --output=count_psds.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abender@ucsd.edu

# Set up environment
cd /home/AD/abender/decoding_spatial_wm
source .env/bin/activate

# Call Python script
python code/count_psds.py