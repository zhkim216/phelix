#!/bin/bash
#SBATCH --job-name=test_eval_fpd
#SBATCH -p possu,bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%A_%a.err
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

# Define directories
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
OUT_DIR=/scratch/users/rshuai/allatom_design
MPNN_PARAMS_DIR=/scratch/users/rshuai/model_params/mpnn
FPD_EMBEDDINGS_DIR=/scratch/users/rshuai/datasets/af3_pdb_monomer/fpd_embeddings
ANNOTATION_CSV=/scratch/users/rshuai/datasets/af3_pdb_monomer/pdb_key_annotations_for_eval2/annotated_pdb_keys.csv

# Model name to evaluate
MODEL_NAME=u1_a2
EXP_NAME=${MODEL_NAME}_eval_fpd_training
WANDB_ID=null
GROUP=eval_fpd_training

# Set HF variables
export HUGGINGFACE_TOKEN=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
export HF_HOME=/scratch/users/rshuai/cache/huggingface

# Directory where denoiser was trained to
DENOISER_TRAIN_DIR=/scratch/users/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_fpd_training.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_PER_TASK} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    fpd_embeddings_dir=${FPD_EMBEDDINGS_DIR} \
    annotation_csv=${ANNOTATION_CSV} \
    use_esm3=true \
    start_step=0 \
    eval_every_n_ckpts=1 \
    batch_size=64 \
    num_steps=500 \
    pct_subsample_train=0.1 \
    max_scrmsd=2.0 \
    max_rel_rog=1.4 \
    subset_length_range=[32,256] \
    hydra.run.dir=${HYDRA_DIR}
