#!/bin/bash
#SBATCH --job-name=lanm_scaffolded_sweep1
#SBATCH -p bioe,possu
#SBATCH --array=0-447%6
#
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
SOFTWARE_PATH=/oak/stanford/groups/possu/rshuai/software # e.g. for MASTER, redis
export SOFTWARE_PATH=${SOFTWARE_PATH} # required for mmseqs, foldseek, redis, etc.

# Model params (if running self-consistency eval)
STRUCT_PRED_MODEL=boltz1
BOLTZ1_CKPT=/scratch/users/rshuai/model_params/boltz1/boltz1_conf.ckpt

# --- Dynamic Inputs Config from Manifest ---
# Define the manifest file containing all job configurations
MANIFEST_FILE="/scratch/users/rshuai/datasets/lanm/lanm_pd_scaffolded/lanm_scaffolded_fixed_pos_manifest.txt"

# Read the line corresponding to the Slurm array task ID
# Since the array is 0-indexed, we need to read the (ID + 1)-th line
JOB_CONFIG_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${MANIFEST_FILE})

# Parse the line to get the PDB directory and the fixed positions
PDB_DIR=$(echo "${JOB_CONFIG_LINE}" | cut -f1)
PDB_DIR="/scratch/users/rshuai/datasets/lanm/lanm_pd_scaffolded/${PDB_DIR}"
FIXED_POS=$(echo "${JOB_CONFIG_LINE}" | cut -f2)
# --- End Dynamic Inputs ---

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Model to use
CKPT_PATH=/scratch/users/rshuai/allatom_design/allatom_design/sd3_a8_S/checkpoints/ema/sd-step25000-epoch15-ema0.99.ckpt
BASE_OUT_DIR=/scratch/users/rshuai/allatom_design/allatom_design/sd3_a8_S/lanm_scaffolded
EXP_NAME=sd3_a8_S_lanm_scaffolded_sweep1_idx${SLURM_ARRAY_TASK_ID}
OUT_DIR=${BASE_OUT_DIR}/${EXP_NAME}

# Log the specific parameters for this job for easier debugging
echo "--- Starting Job ---"
echo "SLURM Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Input PDB Dir: ${PDB_DIR}"
echo "Fixed Positions: ${FIXED_POS}"
echo "Output Dir: ${OUT_DIR}"
echo "--------------------"

# Run training script
cd ${BASE_DIR}
python3 allatom_design/eval/sampling/seq_des_single_multistate.py \
    ckpt_path=${CKPT_PATH} \
    pdb_dir=${PDB_DIR} \
    fixed_pos_seq=\"${FIXED_POS}\" \
    out_dir=${OUT_DIR} \
    struct_pred_cfg.model_name=${STRUCT_PRED_MODEL} \
    struct_pred_cfg.boltz1.checkpoint=${BOLTZ1_CKPT} \
    run_self_consistency_eval=true \
    max_num_conformers=16 \
    sampling_cfg_overrides.batch_size=1 \
    sampling_cfg_overrides.num_seqs_per_pdb=8 \
    +sampling_cfg_overrides.potts_sampling_cfg.potts_sweeps=500 \
    +sampling_cfg_overrides.potts_sampling_cfg.regularization=LCP \
    +sampling_cfg_overrides.omit_aas=['C'] \
    hydra.run.dir=${HYDRA_DIR}