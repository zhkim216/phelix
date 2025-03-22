#!/bin/bash
#SBATCH --job-name=baseline_fs256_eval
#SBATCH -p bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and conda environment
source ~/.subhang_bash_profile
ml python
module load gcc/12.4.0
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh
conda activate allatom_design

# Set base directories
BASE_DIR=/home/users/rshuai/subhang/allatom-design
HYDRA_DIR=/scratch/users/rshuai/subhang/hydra_outputs
OUT_DIR=/scratch/users/rshuai/subhang/allatom_design

# Model params
MPNN_PARAMS_DIR=/scratch/users/rshuai/model_params/mpnn
AF2_DATA_DIR=/scratch/users/rshuai/model_params/af2

# Dataset paths
NNTM_DATASET=/scratch/users/rshuai/datasets/ingraham_cath_dataset/pdb_store_renumbered
DSSP_CSV=/scratch/users/rshuai/datasets/ingraham_cath_dataset/dssp/dssp_length_strict.csv

# Set cache directories
export HF_HOME=/scratch/users/rshuai/subhang/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/subhang/cache/torch
OMEGAFOLD_CACHE=/scratch/users/rshuai/subhang/cache/omegafold/omegafold.pt

# Denoiser checkpoint
MODEL_NAME=config_g_adaln_orig_fs256
AD_CKPT=/scratch/users/rshuai/subhang/allatom_design/allatom_design/${MODEL_NAME}/checkpoints/ad-epoch79.ckpt

# Wandb settings / experiment name
WANDB_ID=null
EXP_NAME=${MODEL_NAME}_eval
GROUP=${MODEL_NAME}_s0

# Run eval script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/bb_unconditional.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    ad_ckpt=${AD_CKPT} \
    num_workers=8 \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    nntm_dataset=${NNTM_DATASET} \
    ss_kld.dssp_csv=${DSSP_CSV} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    +cond_labels.designability=DESIGNABLE \
    hydra.run.dir=${HYDRA_DIR}
