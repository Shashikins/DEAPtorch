#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error
#SBATCH --time=0-01:59:00 # Requested time to run the job
#SBATCH -n 1   # Number of concurrent jobs
#SBATCH -c 64  # Number of cores per task
#SBATCH --mem 16G # Amount of RAM
#SBATCH --gres=gpu:2 # Number of GPUs

# Activate mytorchdist environment.
source $STORE/Cesga2023Courses/pytorch_dist/scripts/activateconda.sh

# Run the project.
srun python main.py
