#!/bin/bash
#SBATCH --job-name=fampnn_ingraham_cath_dataset_residx_fixed_nl0
#SBATCH -p bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%A_%a.err
#SBATCH --array=0-19%10

scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and conda environment
ml python
module load gcc/12.4.0
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh
conda activate allatom_design

# Set base directories
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/users/rshuai/model_params/af2

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Set paths
NOISE_LABELS=0.0
MODEL_CKPT=/scratch/users/rshuai/allatom_design/allatom_design/c2_aasd2/post_hoc_ema_ckpts/ema-step300000-std0.220.ckpt
PDB_DIR=/scratch/users/tianyulu/datasets/ingraham_cath_dataset_residx_fixed/pdb_store
PDB_KEY_LIST=null
OUT_DIR=/scratch/users/rshuai/allatom_design/allatom_design/c2_aasd2/sample_multi/ingraham_cath_dataset_residx_fixed_fampnn32_nl0.0_S100

cd ${BASE_DIR}

# Run the Python script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/sample_multi.py \
    num_workers=${SLURM_CPUS_ON_NODE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    hydra.run.dir=${HYDRA_DIR} \
    checkpoint_path=${MODEL_CKPT} \
    pdb_dir=${PDB_DIR} \
    pdb_key_list=${PDB_KEY_LIST} \
    presort_by_length=true \
    out_dir=${OUT_DIR} \
    num_steps=100 \
    noise_labels=${NOISE_LABELS} \
    temperature=0.1 \
    num_seqs_per_pdb=32 \
    array_id=${SLURM_ARRAY_TASK_ID} \
    num_arrays=20

