#!/bin/bash
#SBATCH --job-name=ligandmpnn_interface_eval
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
MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Input PDB info
PDB_DIR=/scratch/hielab/talal/datasets/af3_pdb/test_pdbs
PDB_KEY_LIST=/scratch/hielab/talal/datasets/af3_pdb/test_pdb_keys_L16_9999_interfaces.list
FIXED_POS_DIR=/scratch/hielab/talal/datasets/af3_pdb/test_interfaces_fixed_pos

# Output base directory for storing results
OUT_BASE=/scratch/hielab/rshuai/allatom_design/allatom_design/interface_evals/proteinmpnn

# Fractions we want to test
FRACTIONS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Each array index 0..19 picks a (method, fraction):
#   Indices 0..9:   method="scn", fraction=FRACTIONS[idx]
#   Indices 10..19: method="seq", fraction=FRACTIONS[idx-10]
IDX=${SLURM_ARRAY_TASK_ID}

if [ $IDX -lt 10 ]; then
    METHOD="scn"
    FRAC_IDX=$IDX
else
    METHOD="seq"
    FRAC_IDX=$((IDX - 10))
fi

T_SEQ=${FRACTIONS[$FRAC_IDX]}

# The CSV file is named fixed_positions_{METHOD}_{T_SEQ}.csv
# e.g. fixed_positions_scn_0.1.csv or fixed_positions_seq_0.1.csv
FIXED_CSV=${FIXED_POS_DIR}/fixed_positions_${METHOD}_${T_SEQ}.csv

mkdir -p ${OUT_BASE}
cd ${BASE_DIR}

# Run the script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/ligandmpnn_interface_eval.py \
    pdb_dir=${PDB_DIR} \
    pdb_key_list=${PDB_KEY_LIST} \
    fixed_pos_csv=${FIXED_CSV} \
    out_base=${OUT_BASE} \
    method=${METHOD} \
    fraction=${T_SEQ} \
    run_ligandmpnn=true \
    ligandmpnn_base_dir=/home/talal/rshuai/other/LigandMPNN \
    mpnn.model_type=protein_mpnn \
    mpnn.checkpoint_name=proteinmpnn_v_48_030.pt \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    mpnn.seed=111

echo "Done with sub-job $IDX"
