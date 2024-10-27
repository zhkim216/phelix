#!/bin/bash
#SBATCH --job-name=eval_fitness_array
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/talal/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/talal/slurm_log_files/job_%j.err

MODEL_NAME=aasd_scn_atom_0.05 /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch ++exp_name=aasd_scn_atom_0.05_fitness_single ++eval_every_n_ckpts=1 ++scoring_method=single
MODEL_NAME=aasd_scn_atom_0.1 /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch ++exp_name=aasd_scn_atom_0.1_fitness_single ++eval_every_n_ckpts=1 ++scoring_method=single
MODEL_NAME=aasd_scn_atom_0.2 /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch ++exp_name=aasd_scn_atom_0.2_fitness_single ++eval_every_n_ckpts=1 ++scoring_method=single
MODEL_NAME=aasd_scn_atom_0.3 /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch ++exp_name=aasd_scn_atom_0.3_fitness_single ++eval_every_n_ckpts=1 ++scoring_method=single
MODEL_NAME=aasd_scn_atom_0.4 /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch ++exp_name=aasd_scn_atom_0.4_fitness_single ++eval_every_n_ckpts=1 ++scoring_method=single
MODEL_NAME=aasd_scn_atom /home/talal/allatom-design/chimera_scripts/talal/eval_fitness_training.sbatch ++exp_name=aasd_scn_atom_fitness_single ++eval_every_n_ckpts=1 ++scoring_method=single
