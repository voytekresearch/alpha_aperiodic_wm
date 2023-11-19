#!/bin/bash

#SBATCH --job-name=spec_param_iem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=68
#SBATCH --mem-per-cpu=2G
#SBATCH --output=spec_param_iem.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abender@ucsd.edu

# Set up environment
cd /home/AD/abender/decoding_spatial_wm
source .env/bin/activate

# Call Python script
python code/spec_param_iem.py