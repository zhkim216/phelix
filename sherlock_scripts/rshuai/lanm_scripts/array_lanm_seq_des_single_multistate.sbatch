#!/bin/bash
#SBATCH --job-name=lanm_sweep1
#SBATCH -p bioe,possu
#SBATCH --array=0-32%6
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

# Inputs config
PDB_DIRS=(
# 0: native
"/scratch/users/rshuai/datasets/lanm/native"

# 1-16: reres
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind120/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind180/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind90/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind120/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind180/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind90/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind120/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind180/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind90/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind120/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind180/8FNS_A_apo_altloc_A_atom_reres"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind90/8FNS_A_apo_altloc_A_atom_reres"

# 17-32: no reres
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind120/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind180/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind90/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind120/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind180/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/bb81-epoch450-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind90/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind120/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind180/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.0-schurn0-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind90/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind120/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind180/8fnr_A_altloc"
"/scratch/users/rshuai/datasets/lanm/lanm_pd/cc89-epoch206-sampling_partial_diffusion-autoguidance-None-ss1.2-schurn200-reconcustom10.0-cfg1.0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind90/8fnr_A_altloc"
)
PDB_DIR=${PDB_DIRS[$SLURM_ARRAY_TASK_ID]}

# FIXED_POS=null
if [ "$SLURM_ARRAY_TASK_ID" -le 16 ]; then
  FIXED_POS="A31-42\,A56-67"
else
  FIXED_POS="A35-46\,A60-71"
fi

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Model to use
CKPT_PATH=/scratch/users/rshuai/allatom_design/allatom_design/sd3_a8_S/checkpoints/ema/sd-step25000-epoch15-ema0.99.ckpt
BASE_OUT_DIR=/scratch/users/rshuai/allatom_design/allatom_design/sd3_a8_S/lanm
EXP_NAME=sd3_a8_S_lanm_sweep1_idx${SLURM_ARRAY_TASK_ID}
OUT_DIR=${BASE_OUT_DIR}/${EXP_NAME}

# Run training script
cd ${BASE_DIR}
python3 allatom_design/eval/sampling/seq_des_single_multistate.py \
    ckpt_path=${CKPT_PATH} \
    pdb_dir=${PDB_DIR} \
    fixed_pos_seq=${FIXED_POS} \
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