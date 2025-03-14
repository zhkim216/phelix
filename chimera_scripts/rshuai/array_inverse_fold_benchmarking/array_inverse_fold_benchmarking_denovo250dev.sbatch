#!/bin/bash
#SBATCH --job-name=if_benchmark
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#SBATCH --array=0-24%5

# Define the list of PDB directories to evaluate
EPS_VALUES=(0.0 0.1 0.2 0.3 0.4)
NOISE_LABELS_VALUES=(0.0 0.1 0.2 0.3 0.4)
INDEX=${SLURM_ARRAY_TASK_ID}
EPS_INDEX=$((INDEX / 5))
NOISE_INDEX=$((INDEX % 5))
EPS=${EPS_VALUES[$EPS_INDEX]}
NOISE_LABELS=${NOISE_LABELS_VALUES[$NOISE_INDEX]}

# We'll use a separate variable for the PDB directory stem
PDB_DIR_STEM="denovo250dev_S100_so_nl${NOISE_LABELS}"

# Construct the EXP_NAME to reflect both model name and the noise label
EXP_NAME="${PDB_DIR_STEM}_c1_aasd2_eps${EPS}"

# Construct the actual PDB_DIR path
PDB_DIR="c1_aasd2_eps${EPS}/sample_multi/sd-step150000-epoch450/${PDB_DIR_STEM}/samples"

if [ -z "${PDB_DIR}" ]; then
  echo "Error: PDB_DIR is empty."
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

# Set input PDB directory
INPUT_PDB_DIR="/scratch/hielab/rshuai/allatom_design/allatom_design/${PDB_DIR}"
OUT_DIR="/scratch/hielab/rshuai/allatom_design/allatom_design/c1_aasd2_eps${EPS}/inverse_fold_benchmarking/sd-step150000-epoch450/${PDB_DIR_STEM}"

# Run inverse folding benchmarking for the current pdb dir
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/inverse_fold_benchmarking.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=8 \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    sample_dir=/scratch/hielab/rshuai/datasets/denovo250dev/pdbs \
    hydra.run.dir=${HYDRA_DIR} \
    input_pdb_dir=${INPUT_PDB_DIR} \
    out_dir=${OUT_DIR}
