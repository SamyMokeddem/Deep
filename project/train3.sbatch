#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=Weather3           # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=train3.log   # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=24:00:00              # Max execution time

python weather_diffusion.py --train --auto_name --pred_step=5 --T=50 