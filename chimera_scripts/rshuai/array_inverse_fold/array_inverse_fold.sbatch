#!/bin/bash
#SBATCH --job-name=b3_aasd5_inv_fold
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0-26%14
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and activate conda environment
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/rshuai/cache/omegafold/omegafold.pt

# Sweep parameters
temperature_values=(0.0 0.1 0.2)
num_corrector_steps_values=(0 1 5)
corrector_step_ratio_values=(0.0 0.1 0.2 0.5)

# Calculate indices
temp_idx=$(( SLURM_ARRAY_TASK_ID / 9 ))  # 27 / 3 = 9
rem=$(( SLURM_ARRAY_TASK_ID % 9 ))

if [ $rem -eq 0 ]; then
    nc=0
    csr=0.0
elif [ $rem -ge 1 ] && [ $rem -le 4 ]; then
    nc=1
    csr=${corrector_step_ratio_values[$((rem - 1))]}
elif [ $rem -ge 5 ] && [ $rem -le 8 ]; then
    nc=5
    csr=${corrector_step_ratio_values[$((rem - 5))]}
fi

# Assign temperature
tau=${temperature_values[$temp_idx]}

# Denoiser checkpoint
MODEL_NAME=b3_aasd5
SD_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/checkpoints/sd-step200000-epoch600.ckpt

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_inv_fold_greedy_tau${tau}_S20_nc${nc}_csr${csr}
GROUP=${MODEL_NAME}_inv_fold2_greedy_corrector

# Run inverse folding script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/inverse_fold.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    sd_ckpt=${SD_CKPT} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    temperature=${tau} \
    timestep_schedule.num_steps=20 \
    aatype_decoding_order_mode=greedy \
    num_corrector_steps=${nc} \
    corrector_step_ratio=${csr} \
    num_pdbs=300 \
    num_seqs_per_pdb=1 \
    batch_size=64 \
    hydra.run.dir=${HYDRA_DIR}
