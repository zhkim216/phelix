#!/bin/bash
#SBATCH --job-name=metadata_atomworks_pdb_full_v8
#SBATCH -p possu,normal,owners
#SBATCH --array=0-99
#SBATCH --cpus-per-task=8
#SBATCH --time=1-0:00:00
#SBATCH --mem=32GB

# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/build_metadata_parquet_shards_v8_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/build_metadata_parquet_shards_v8_%A_%a.err

## Load modules and set environment variables:
BASE_DIR=/home/users/zhkim216/code/allatom-design

export PDB_MIRROR_PATH="/scratch/users/zhkim216/datasets/pdb_mirror"
export CCD_MIRROR_PATH="/scratch/users/zhkim216/datasets/ccd_mirror"

# Create output directories if they don't exist
mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing

cd ${BASE_DIR}

# Calculate num_shards from SLURM_ARRAY_TASK_MAX (extracted from --array directive)
NUM_SHARDS=$((SLURM_ARRAY_TASK_MAX + 1))

echo "Starting shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} for atomworks_pdb_full_v8"

# Run script
# Key changes from v7:
#   - contact_distance: 5.0 -> 8.0 (captures partners up to 8A, filtered downstream by min_distance)
#   - metal_coordination_cfg: NEW (counts coordination partners for metals)
#   - small_molecule_neighboring_distance: NEW (counts heavy atoms from other PN units within 5A)
#   - New parquet columns: q_pn_unit_n_coordination_partners_metal, q_pn_unit_n_neighboring_heavy_atoms_small_molecule, q_pn_unit_avg_occupancy_nonpolymer
python3 allatom_design/data/preprocessing/atomworks/build_metadata_parquet_shards.py \
    --config-path=/home/users/zhkim216/code/allatom-design/allatom_design/configs/data/preprocessing/atomworks \
    --config-name=build_metadata_parquet_shards \
    mmcif_dir=/scratch/users/zhkim216/datasets/pdb_mirror \
    out_dir=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v8 \
    num_workers=${SLURM_CPUS_PER_TASK} \
    num_shards=${NUM_SHARDS} \
    shard_id=${SLURM_ARRAY_TASK_ID} \
    max_file_size=5000000 \
    fetch_ligand_validity_scores=false \
    data_preprocessor_cfg.polymer_pn_unit_limit=500 \
    data_preprocessor_cfg.clash_distance=1.0 \
    data_preprocessor_cfg.contact_distance=8.0 \
    data_preprocessor_cfg.close_distance=30.0 \
    data_preprocessor_cfg.from_rcsb=true \
    data_preprocessor_cfg.small_molecule_neighboring_distance=5.0 \
    data_preprocessor_cfg.metal_coordination_cfg.coordination_distance=3.2 \
    processing_timeout=600 \
    batch_size=200 \
    max_tasks_per_child=25

echo "Completed shard ${SLURM_ARRAY_TASK_ID}"
