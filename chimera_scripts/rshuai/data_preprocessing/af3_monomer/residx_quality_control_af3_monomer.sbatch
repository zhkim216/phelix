#!/bin/bash
#SBATCH --job-name=resid_quality_control
#SBATCH -p cpu_batch
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design
BASE_DIR=/home/talal/rshuai/allatom-design

cd ${BASE_DIR}

# Run script
PHASE=eval
python3 allatom_design/data/preprocessing/residx_quality_control_af3_monomer.py \
    /scratch/hielab/rshuai/datasets/af3_pdb_monomer/${PHASE}_pdb_keys_all.list \
    --mmcif-dir /large_storage/hielab/rshuai/datasets/mmCIF \
    --af3-mmcif-dir /scratch/hielab/talal/datasets/af3_pdb/${PHASE}_mmcifs \
    --save-dir /scratch/hielab/rshuai/datasets/af3_pdb_monomer/${PHASE}_mmcifs \
    --out-list-name ${PHASE}_pdb_keys.list \
    --max-threads ${SLURM_CPUS_PER_TASK}
