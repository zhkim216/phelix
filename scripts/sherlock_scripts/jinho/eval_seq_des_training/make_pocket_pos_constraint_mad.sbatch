#!/bin/bash
#SBATCH --job-name=make_pocket_constraint_mad
#SBATCH -p possu,bioe
#SBATCH --array=2-4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/make_pocket_constraint_mad_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/make_pocket_constraint_mad_%A_%a.err

# ==============================================================================
# Make pocket positional constraint CSVs using per-PDB MAD thresholding.
# Pocket residues (delta_J >= threshold) go into fixed_pos_seq (= fixed);
# scaffold residues (delta_J < threshold) are resampled downstream.
#
# Input: exp35_cfg2 step82500 stage-1 samples under
#   .../step_82500_epoch_52/samples/{base}_sample{s}.cif
# NUM_SAMPLES controls how many per-base stage-1 variants to forward-pass
# (NUM_SAMPLES=1 → only `_sample0` per base, one row per base).
#
# Output row schema: pdb_key = Path(cif).stem (includes `_sample{s}`),
#   fixed_pos_seq, fixed_pos_scn, num_{scaffold,pocket}_residues, num_tokens.
# Output filename: pocket_constraint_mad_k{K}_numsample{N}_array_{C}.csv
#   (trailing `_array_{C}` matches the convention expected by
#    scripts/sherlock_scripts/jinho/utils/merge_array_csvs.py)
#
# Method: delta_J per-PDB MAD, k = 0.2, 0.4, 0.6, 0.8, 1.0, 1.2
# Array: 5 chunks (--array=0-4); each processes a shard × all k values
# ==============================================================================
NUM_CHUNKS=5
NUM_SAMPLES=1   # per-base stage-1 sample count to forward-pass (1 = _sample0 only)

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUTPUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/pocket_pos_constraint_mad

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${OUTPUT_DIR}"

cd "${BASE_DIR}"

export PYTHONWARNINGS="ignore"

# Model checkpoint
CKPT_PATH=/scratch/users/zhkim216/out_dir/ligand_seq_des/exp35_cfg2/checkpoints/ema/sd-step82500-epoch52-ema0.99.ckpt

# Input: stage-1 sample CIFs from exp35_cfg2 step82500
PDB_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg2_denovoval_full_af3/step_82500_epoch_52/samples

echo "Starting pocket constraint generation (per-PDB MAD)"
echo "  Chunk: ${SLURM_ARRAY_TASK_ID}/${NUM_CHUNKS}"
echo "  Per-base sample count: ${NUM_SAMPLES}"
echo "  CIF dir: ${PDB_DIR}"
echo "  Output: ${OUTPUT_DIR}"

python3 -m allatom_design.eval.sampling.make_pocket_pos_constraint_mad \
    debug=false \
    seed=0 \
    ckpt_path=${CKPT_PATH} \
    output_dir=${OUTPUT_DIR} \
    'k_values=[0.2,0.4,0.6,0.8,1.0,1.2]' \
    num_samples=${NUM_SAMPLES} \
    max_tokens_for_forward=2000 \
    input_sample_is_designed=true \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.array_id=${SLURM_ARRAY_TASK_ID} \
    pdb_cfg.num_arrays=${NUM_CHUNKS} \
    featurizer_cfg.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    hydra.run.dir=${HYDRA_DIR}

echo "Done chunk ${SLURM_ARRAY_TASK_ID}/${NUM_CHUNKS}"
