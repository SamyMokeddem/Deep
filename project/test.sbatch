#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=WeatherTest           # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=test.log   # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=24:00:00              # Max execution time

python weather_diffusion.py --test --auto_name --model_path="train_models/T50_PS3_lr0.0001_B64_E5_A_final.pth" --batch_size=64 --pred_step=3 --attention --ensemble=5 --T=50