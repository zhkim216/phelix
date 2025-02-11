#!/bin/bash
#SBATCH --job-name=sample_single
#SBATCH -p bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%j.out  # CHANGE ME
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%j.err  # CHANGE ME
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh  # CHANGE ME (or delete me)

## Load modules and conda environment
ml python
module load gcc/12.4.0
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh
conda activate allatom_design

# Set base directories
BASE_DIR=/home/users/rshuai/code/allatom-design  # CHANGE ME
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs  # CHANGE ME

# Model params
AF2_DATA_DIR=/scratch/users/rshuai/model_params/af2

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface  # CHANGE ME
export TORCH_HOME=/scratch/users/rshuai/cache/torch  # CHANGE ME

# Denoiser checkpoint
MODEL_CKPT=/scratch/users/rshuai/allatom_design/allatom_design/b12_aasd6/checkpoints/sd-step580000-epoch345.ckpt

# Run eval script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/sample_single.py \
    num_workers=${SLURM_CPUS_ON_NODE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    hydra.run.dir=${HYDRA_DIR} \
    checkpoint_path=${MODEL_CKPT} \
    batch_size=16 \
    pdb_path=/scratch/users/rshuai/datasets/rcsb_test_cases/pdbs/1TIM.pdb \
    fixed_pos_seq=\'A1-100,B1-100\' \
    fixed_pos_scn=\'A1-100\' \
    num_steps=100 \
    num_samples=32 \
    temperature=0.1 \
    run_self_consistency_eval=false \
    out_dir=/scratch/users/rshuai/allatom_design/allatom_design/test_sample_single
