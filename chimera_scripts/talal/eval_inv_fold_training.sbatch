#!/bin/bash
#SBATCH --job-name=eval_if
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/talal/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/talal/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Set base directories
BASE_DIR=/home/talal/allatom-design
HYDRA_DIR=/scratch/hielab/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Dataset paths
PDB_PATH=/scratch/hielab/talal/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/hielab/talal/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv

# Set cache directories
export HF_HOME=/scratch/hielab/talal/cache/huggingface
export TORCH_HOME=/scratch/hielab/talal/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/talal/cache/omegafold/omegafold.pt

# Model to evaluate
MODEL_NAME=aasd_gt_atom_decoder
DENOISER_TRAIN_DIR=/scratch/hielab/talal/allatom_design/allatom_design/${MODEL_NAME}

# Wandb settings / experiment name
WANDB_ID=talal-widatalla
EXP_NAME=${MODEL_NAME}_eval_inv_fold
GROUP=chimera_a7

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_inv_fold_training.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    start_epoch=100 \
    aatype_decoding_order_mode=greedy \
    num_steps_list=[1,3,5,10] \
    eval_every_n_ckpts=4 \
    batch_size=64 \
    run_seq_recovery=true \
    run_codes_sc=false \
    num_codes_sc_batches=1 \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    subset_length_range=[64,256] \
    hydra.run.dir=${HYDRA_DIR}