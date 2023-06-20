#!/bin/bash

#SBATCH --job-name=spec_param_iem_cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=spec_param_iem_cluster.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abender@ucsd.edu

# Set up environment
cd /home/AD/abender/decoding_spatial_wm
source .env/bin/activate

# Call Python script
python code/spec_param_iem.py