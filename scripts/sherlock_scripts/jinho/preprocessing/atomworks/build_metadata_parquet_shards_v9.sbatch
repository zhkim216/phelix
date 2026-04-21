#!/bin/bash
#SBATCH --job-name=metadata_atomworks_pdb_full_v9
#SBATCH -p possu,normal,owners
#SBATCH --array=0-99
#SBATCH --cpus-per-task=8
#SBATCH --time=1-0:00:00
#SBATCH --mem=128GB

# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/build_metadata_parquet_shards_v9_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/build_metadata_parquet_shards_v9_%A_%a.err

## Load modules and set environment variables:
BASE_DIR=/home/users/zhkim216/code/allatom-design

export PDB_MIRROR_PATH="/scratch/users/zhkim216/datasets/pdb_mirror"
export CCD_MIRROR_PATH="/scratch/users/zhkim216/datasets/ccd_mirror"

# Create output directories if they don't exist
mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing

cd ${BASE_DIR}

# Calculate num_shards from SLURM_ARRAY_TASK_MAX (extracted from --array directive)
NUM_SHARDS=$((SLURM_ARRAY_TASK_MAX + 1))

echo "Starting shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} for atomworks_pdb_full_v9"

# Run script
# Key changes from v8:
#   - halide_coordination_cfg: NEW. Counts non-C atoms within 5A of single-atom halide ions
#     (F/CL/BR/IOD). partner_mask = (element != 'C'), modality-agnostic.
#   - metal_coordination_cfg.min_partner_occupancy: REMOVED. occupancy>0 is already enforced
#     upstream by DataPreprocessor._apply_filters, so this knob was dead weight.
#   - Per-partner contact breakdown for non-polymer query rows (metal / halide / small molecule).
#     Each row gets one JSON-stringified list of {pn_unit_iid, chain_iid, count} for the
#     applicable type, with the other two fields set to None.
#
# New parquet columns (added on top of v8 schema):
#   q_pn_unit_is_halide                                (bool)
#   q_pn_unit_n_coordination_partners_halide           (int | None)   [halide rows only]
#   q_pn_unit_per_partner_contacts_metal               (JSON str | None) [donor-only, 3.2A]
#   q_pn_unit_per_partner_contacts_halide              (JSON str | None) [non-C,     5.0A]
#   q_pn_unit_per_partner_contacts_small_molecule      (JSON str | None) [all atoms, 5.0A]
#
# Pipeline fix bundled with v9:
#   _add_parquet_columns now explicitly casts the 4 numeric-nullable columns
#   (n_coordination_partners_{metal,halide}, n_neighboring_heavy_atoms_small_molecule,
#    avg_occupancy_nonpolymer) with pd.to_numeric(errors='coerce') BEFORE the blanket
#   object->str pass, so batches with all-None values no longer stringify to "None" and
#   break pyarrow type inference at merge time.
python3 allatom_design/data/preprocessing/atomworks/build_metadata_parquet_shards.py \
    --config-path=/home/users/zhkim216/code/allatom-design/allatom_design/configs/data/preprocessing/atomworks \
    --config-name=build_metadata_parquet_shards \
    mmcif_dir=/scratch/users/zhkim216/datasets/pdb_mirror \
    out_dir=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9 \
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
    data_preprocessor_cfg.halide_coordination_cfg.coordination_distance=5.0 \
    processing_timeout=600 \
    batch_size=200 \
    max_tasks_per_child=25

echo "Completed shard ${SLURM_ARRAY_TASK_ID}"
