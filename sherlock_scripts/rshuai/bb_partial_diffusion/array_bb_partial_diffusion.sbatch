#!/bin/bash
#SBATCH --job-name=bb_partial_diffusion
#SBATCH -p bioe
#SBATCH --array=0-4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%A_%a.err
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and set environment variables
ml python
module load gcc/12.4.0
module load cuda/12.4
ENV_DIR=/home/groups/possu/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate

# Define directories and paths
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
SOFTWARE_PATH=/oak/stanford/groups/possu/rshuai/software  # e.g. for MASTER, redis
export SOFTWARE_PATH=${SOFTWARE_PATH}  # required for mmseqs, foldseek, redis, etc.

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Inputs config
# PDB_DIR=/scratch/users/rshuai/datasets/bindcraft_traj_benchmark/pdbs
# PDB_NAME_LIST=/scratch/users/rshuai/datasets/bindcraft_traj_benchmark/pdb_name_lists/subset_20.txt
PDB_DIR=/scratch/users/rshuai/datasets/boltz_test_cifs/pdbs
PDB_NAME_LIST=/scratch/users/rshuai/datasets/boltz_test_cifs/pdb_name_lists/protein_monomer_32_512_no_ligand.txt

# Denoiser checkpoint
CKPT_PATH=/scratch/users/rshuai/allatom_design/allatom_design/spd0_a4_cfg2/checkpoints/ema/ad-step625000-epoch624-ema0.99.ckpt

# Outputs config
# BASE_OUT_DIR=/scratch/users/rshuai/datasets/bindcraft_traj_benchmark/conformers/spd0_a4_cfg2
BASE_OUT_DIR=/scratch/users/rshuai/datasets/boltz_test_cifs/conformers/spd0_a4_cfg2

# Sweep params
T_VALS=(
"0.6"
"0.65"
"0.7"
"0.75"
"0.8"
"0.85"
"0.9"
)
NUM_STEPS_VALS=(
    "200"
    "175"
    "150"
    "125"
    "100"
    "75"
    "50"
)
T_VAL=${T_VALS[$SLURM_ARRAY_TASK_ID]}
NUM_STEPS=${NUM_STEPS_VALS[$SLURM_ARRAY_TASK_ID]}
EXP_NAME=t${T_VAL}_n${NUM_STEPS}

# Run script
cd ${BASE_DIR}
python3 allatom_design/eval/sampling/bb_partial_diffusion.py \
    ckpt_path=${CKPT_PATH} \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.pdb_name_list=${PDB_NAME_LIST} \
    base_out_dir=${BASE_OUT_DIR} \
    exp_name=${EXP_NAME} \
    bb_gen_cfg.overrides.timestep_schedule.t_start=${T_VAL} \
    bb_gen_cfg.overrides.num_steps=${NUM_STEPS} \
    bb_gen_cfg.overrides.batch_size=16 \
    n_samples_per_pdb=32 \
    hydra.run.dir=${HYDRA_DIR}
