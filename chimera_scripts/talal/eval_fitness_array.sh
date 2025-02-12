#!/bin/bash
#SBATCH --job-name=eval_fitness_array
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=98:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/talal/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/talal/slurm_log_files/job_%j.err

MODEL_NAME=aasd_pdb EXP_NAME=${MODEL_NAME}_fitness /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch
MODEL_NAME=aasd_gvp_pdb EXP_NAME=${MODEL_NAME}_fitness /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch 
MODEL_NAME=aasd_gvp_trans_pdb EXP_NAME=${MODEL_NAME}_fitness /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch 
MODEL_NAME=aasd_sidechain_pdb EXP_NAME=${MODEL_NAME}_fitness /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch
MODEL_NAME=aasd_gvp_pdb_0.3 EXP_NAME=${MODEL_NAME}_fitness /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch 
MODEL_NAME=aasd_gvp_trans_pdb_0.3 EXP_NAME=${MODEL_NAME}_fitness /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch 
MODEL_NAME=aasd_sidechain_pdb_0.3 EXP_NAME=${MODEL_NAME}_fitness /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch 
MODEL_NAME=aasd_pdb_0.3 EXP_NAME=${MODEL_NAME}_fitness /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch 
