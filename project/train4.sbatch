#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=Weather8           # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=train8.log   # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=24:00:00              # Max execution time

python weather_diffusion.py --train --auto_name --batch_size=64 --num_epochs=300 --pred_step=3 --lr=0.0001 --ensemble=5 --dropout=0.2 --T=50 