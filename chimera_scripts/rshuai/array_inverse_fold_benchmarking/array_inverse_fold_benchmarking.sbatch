#!/bin/bash
#SBATCH --job-name=if_benchmark
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#SBATCH --array=0-4

# Define the list of FASTA directories to evaluate
# FASTAS=("esm3" "esm_if1" "lmdesign")
# FASTAS=(" "proteinmpnn")
# FASTAS=("S1_c0_aasd2" "S10_c0_aasd2" "S100_c0_aasd2" "S300_c0_aasd2" "S500_c0_aasd2" "S100_seq_only_c0_aasd2" "S100_c0_aasd4")
# FASTAS=("S100_psce0.3_c0_aasd2" "S100_psce0.5_c0_aasd2" "S100_psce1.0_c0_aasd2")
FASTAS=("val_S100_psce0.1_c0_aasd2" "val_S100_psce0.2_c0_aasd2" "val_S100_psce0.3_c0_aasd2" "val_S100_psce0.5_c0_aasd2" "val_S100_psce1.0_c0_aasd2")
FASTA_NAME=${FASTAS[$SLURM_ARRAY_TASK_ID]}

if [ -z "${FASTA_NAME}" ]; then
  echo "Error: FASTA_NAME is empty."
  exit 1
fi

# Load modules and conda environment
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories and parameters
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/rshuai/cache/omegafold/omegafold.pt

WANDB_ID=richardshuai
GROUP=denovo500_if_benchmarking

cd ${BASE_DIR}

# Set input FASTA directory and output directory based on the current FASTA
# INPUT_FASTA_DIR="/scratch/hielab/rshuai/allatom_design/allatom_design/sc_evals/fastas/${FASTA_NAME}"
INPUT_PDB_DIR="/scratch/hielab/rshuai/allatom_design/allatom_design/sc_evals/input_pdbs/${FASTA_NAME}"
OUT_DIR="/scratch/hielab/rshuai/allatom_design/allatom_design/sc_evals/outputs/${FASTA_NAME}"

# Run inverse folding benchmarking for the current FASTA
    # input_fasta_dir=${INPUT_FASTA_DIR} \
    # sample_dir=/scratch/hielab/rshuai/datasets/denovo500/pdbs \
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/inverse_fold_benchmarking.py \
    wandb_id=${WANDB_ID} \
    exp_name=${FASTA_NAME} \
    group=${GROUP} \
    num_workers=8 \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    sample_dir=/scratch/hielab/talal/datasets/af3_pdb/val_pdbs_L64_512 \
    skip_pdbs=["7xva_A_1485"] \
    hydra.run.dir=${HYDRA_DIR} \
    input_pdb_dir=${INPUT_PDB_DIR} \
    out_dir=${OUT_DIR}
