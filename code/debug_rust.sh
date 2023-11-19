#!/bin/bash

#SBATCH --job-name=debug_rust
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2GB
#SBATCH --output=debug_rust.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abender@ucsd.edu

# Set up environment
cd /home/AD/abender/decoding_spatial_wm
source .env/bin/activate

# Call Python script
python code/debug_rust.py