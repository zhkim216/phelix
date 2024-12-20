#!/bin/bash
#SBATCH --job-name=sample_multi_cath
#SBATCH -p possu
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

# Updated paths from the local launch.json configuration
MODEL_CKPT=/scratch/users/rshuai/allatom_design/allatom_design/b17_aasd9/post_hoc_ema_ckpts/ema-step200000-std0.220.ckpt
PDB_DIR=/scratch/users/rshuai/datasets/ingraham_cath_dataset/pdb_store
PDB_KEY_LIST=/scratch/users/rshuai/datasets/ingraham_cath_dataset/train_pdb_keys_L50_150.list
OUT_DIR=/scratch/users/rshuai/allatom_design/allatom_design/b17_aasd9/sample_multi_cath/ingraham_cath_L50_150_S50_T0.5

cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/sample_multi.py \
    num_workers=${SLURM_CPUS_ON_NODE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    hydra.run.dir=${HYDRA_DIR} \
    checkpoint_path=${MODEL_CKPT} \
    pdb_dir=${PDB_DIR} \
    pdb_key_list=${PDB_KEY_LIST} \
    out_dir=${OUT_DIR} \
    run_self_consistency_eval=true \
    num_steps=50 \
    num_seqs_per_pdb=4 \
    temperature=0.5
