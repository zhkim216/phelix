#!/bin/bash
#SBATCH --job-name=score-conformers
#SBATCH -p bioe
#SBATCH --array=0-23%12
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

# Model params (if running self-consistency eval)
STRUCT_PRED_MODEL=boltz1
BOLTZ1_CKPT=/scratch/users/rshuai/model_params/boltz1/boltz1_conf.ckpt

# Backbones we want to score on
CONFORMER_BASE_DIR=/scratch/users/rshuai/datasets/boltz_val_cifs/conformers/protein_monomer_32_512_no_ligand
CONFORMER_NAMES=(
    "gaussian_eps0.3"
    "gaussian_eps0.5"
    "s6_a0_t0.7"
    "aicath_sde_pd90"
)
CONFORMER_DIRS=(
    "gaussian_conformers/eps0.3"
    "gaussian_conformers/eps0.5"
    "s6_a0_conformers/t0.7"
    "md_pd_conformers-sde-aicath-no-recurrence/sde-aicath-no-recurrence-richards-structures-90"
)

# Samples we want to score for each backbone
SAMPLE_BASE_DIR=/scratch/users/rshuai/allatom_design/allatom_design/sd3_a2_cfg1/multistate_sweep
SAMPLE_NAMES=(
    "baseline"
    "baseline_noreg"
    "gaussian_eps0.3"
    "gaussian_eps0.5"
    "s6_a0_t0.7"
    "aicath_sde_pd90"
)
SAMPLE_DIRS=(
    "sd3_a2_cfg1_baseline_ps500_bvc_protein_monomer_32_512_no_ligand_boltz1/samples"
    "sd3_a2_cfg1_baseline_ps5000_noreg_bvc_protein_monomer_32_512_no_ligand_boltz1/samples"
    "sd3_a2_cfg1_ms_sweep8_eps0.3_nc16_bvc_protein_monomer_32_512_no_ligand_boltz1/samples"
    "sd3_a2_cfg1_ms_sweep8_eps0.5_nc16_bvc_protein_monomer_32_512_no_ligand_boltz1/samples"
    "sd3_a2_cfg1_ms_sweep2_t0.7_nc16_bvc_protein_monomer_32_512_no_ligand_boltz1/samples"
    "sd3_a2_cfg1_ms_sweep6_pd90_nc16_bvc_protein_monomer_32_512_no_ligand_boltz1/samples"
)

# Compute indices for Cartesian product
NUM_SAMPLES=6
CONFORMER_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SAMPLES ))
SAMPLE_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SAMPLES ))

CONFORMER_NAME=${CONFORMER_NAMES[${CONFORMER_IDX}]}
CONFORMER_DIR=${CONFORMER_BASE_DIR}/${CONFORMER_DIRS[${CONFORMER_IDX}]}
SAMPLE_NAME=${SAMPLE_NAMES[${SAMPLE_IDX}]}
SAMPLE_DIR=${SAMPLE_BASE_DIR}/${SAMPLE_DIRS[${SAMPLE_IDX}]}

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Model to use
CKPT_PATH=/scratch/users/rshuai/allatom_design/allatom_design/sd3_a2_cfg1/checkpoints/ema/sd-step17500-epoch22-ema0.99.ckpt
BASE_OUT_DIR=/scratch/users/rshuai/allatom_design/allatom_design/sd3_a2_cfg1/score
OUT_DIR=${BASE_OUT_DIR}/${CONFORMER_NAME}/${SAMPLE_NAME}

# Run scoring script
cd ${BASE_DIR}
python3 allatom_design/eval/sampling/score_conformers.py \
    ckpt_path=${CKPT_PATH} \
    conformer_dir=${CONFORMER_DIR} \
    sample_dir=${SAMPLE_DIR} \
    out_dir=${OUT_DIR} \
    hydra.run.dir=${HYDRA_DIR}
