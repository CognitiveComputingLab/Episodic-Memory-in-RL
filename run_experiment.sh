#!/bin/bash

# --- SLURM Settings ---
#SBATCH -N 1                            # Nodes
#SBATCH -c 1                            # Cores
#SBATCH --output=logs/%x_%j.out         # Output log (make sure 'logs' dir exists)
#SBATCH --error=logs/%x_%j.err          # Error log
#SBATCH --partition=ug-gpu-small       # Partition name
#SBATCH --qos=short                     # QOS
#SBATCH --gres=gpu:1g.10gb:1            # GPUs
#SBATCH --mem=8G                        # RAM

# --- Environment Setup ---
module load cuda/12.4
# module load python/3.10
source ~/python_3.11_env/bin/activate

# --- Debugging Info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# --- WandB Setup ---
# If you are not logged in via CLI, uncomment the line below and add your key
# export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"


mkdir -p logs
python3 run_experiment.py "$1"
echo "Job finished at: $(date)"