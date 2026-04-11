#!/bin/bash
#SBATCH --job-name=make_scaffold_constraint_mad
#SBATCH -p possu,bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/make_scaffold_constraint_mad_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/make_scaffold_constraint_mad_%j.err

# ==============================================================================
# Make scaffold positional constraint CSVs using per-PDB MAD thresholding
# Dataset: denovo_val_cifs (4880 CIFs)
# Method: delta_J per-PDB MAD, k = 0.2, 0.6, 1.0, 1.4, 1.8
# ==============================================================================

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUTPUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/scaffold_pos_constraint_mad

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${OUTPUT_DIR}"

cd "${BASE_DIR}"

export PYTHONWARNINGS="ignore"

# Model checkpoint
CKPT_PATH=/scratch/users/zhkim216/out_dir/ligand_seq_des/exp35_cfg2/checkpoints/ema/sd-step82500-epoch52-ema0.99.ckpt

# Input: full denovo val CIF set
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/denovo_val_cifs

echo "Starting scaffold constraint generation (per-PDB MAD)"
echo "  CIF dir: ${PDB_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Checkpoint: ${CKPT_PATH}"

python3 -m allatom_design.eval.sampling.make_scaffold_pos_constraint_mad \
    debug=false \
    seed=0 \
    ckpt_path=${CKPT_PATH} \
    output_dir=${OUTPUT_DIR} \
    'k_values=[0.2,0.6,1.0,1.4,1.8]' \
    num_samples=2 \
    max_tokens_for_forward=2000 \
    input_sample_is_designed=true \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    featurizer_cfg.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    hydra.run.dir=${HYDRA_DIR}

echo "Done. Output saved to ${OUTPUT_DIR}"
