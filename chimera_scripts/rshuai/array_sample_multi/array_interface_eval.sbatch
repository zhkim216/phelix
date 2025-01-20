#!/bin/bash
#SBATCH --job-name=c0_aasd2_interface_eval
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%j_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%j_%a.err
#SBATCH --array=0-19%2

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and conda environment
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Set paths
MODEL_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/c0_aasd2/post_hoc_ema_ckpts/ema-step300000-std0.010.ckpt
PDB_DIR=/scratch/hielab/talal/datasets/af3_pdb/test_pdbs
PDB_KEY_LIST=/scratch/hielab/talal/datasets/af3_pdb/test_pdb_keys_L16_9999_interfaces.list
FIXED_POS_DIR=/scratch/hielab/talal/datasets/af3_pdb/test_interfaces_fixed_pos

# Fractions we want to test
FRACTIONS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Each array index 0..19 picks a (method, fraction).
#  Index in [0..9]:  method="scn", fraction=FRACTIONS[$idx]
#  Index in [10..19]: method="seq", fraction=FRACTIONS[$idx - 10]

IDX=${SLURM_ARRAY_TASK_ID}

if [ $IDX -lt 10 ]; then
    METHOD="scn"
    FRAC_IDX=$IDX
else
    METHOD="seq"
    FRAC_IDX=$((IDX - 10))
fi

T_SEQ=${FRACTIONS[$FRAC_IDX]}

# The CSV file is named fixed_positions_{METHOD}_{T_SEQ}.csv, e.g. fixed_positions_scn_0.1.csv
FIXED_CSV=${FIXED_POS_DIR}/fixed_positions_${METHOD}_${T_SEQ}.csv

# Output directory
# e.g. /scratch/.../interface_eval/scn_0.1
OUT_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/c0_aasd2/sample_multi/ema-step300000-std0.010/interface_eval/${METHOD}_${T_SEQ}

mkdir -p ${OUT_DIR}

# Run script
cd ${BASE_DIR}

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/sample_multi.py \
    num_workers=${SLURM_CPUS_ON_NODE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    hydra.run.dir=${HYDRA_DIR} \
    checkpoint_path=${MODEL_CKPT} \
    pdb_dir=${PDB_DIR} \
    pdb_key_list=${PDB_KEY_LIST} \
    fixed_pos_csv=${FIXED_CSV} \
    pdb_key_ext=.pdb \
    out_dir=${OUT_DIR} \
    repack_last=true \
    seq_only=true \
    presort_by_length=true \
    batch_size=2 \
    num_steps=1 \
    temperature=0.0
