#!/bin/bash
#SBATCH --job-name=resid_quality_control
#SBATCH -p cpu_batch
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0-49%20
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design
BASE_DIR=/home/talal/rshuai/allatom-design

cd ${BASE_DIR}

# Run script
python3 allatom_design/data/preprocessing/af3_pdb_monomer/residx_quality_control_af3_monomer.py \
    /scratch/hielab/rshuai/datasets/af3_pdb_monomer/preprocessing/parse_af3_clustered_chains/all_pdb_keys.csv \
    --mmcif-dir /large_storage/hielab/rshuai/datasets/mmCIF \
    --af3-eval-mmcif-dir /scratch/hielab/talal/datasets/af3_pdb/eval_mmcifs \
    --af3-train-mmcif-dir /scratch/hielab/talal/datasets/af3_pdb/train_mmcifs \
    --save-dir /scratch/hielab/rshuai/datasets/af3_pdb_monomer/preprocessing/residx_quality_control_af3_monomer \
    --max-threads ${SLURM_CPUS_ON_NODE} \
    --array-id ${SLURM_ARRAY_TASK_ID} \
    --num-arrays 50
