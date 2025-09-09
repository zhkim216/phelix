#!/bin/bash
#SBATCH --job-name=annotate_designability_csv
#SBATCH -p bioe,possu
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%A_%a.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
ml python
module load gcc/12.4.0
module load cuda/12.4
ENV_DIR=/home/groups/possu/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate

BASE_DIR=/home/users/rshuai/code/allatom-design
cd ${BASE_DIR}
python3 allatom_design/data/preprocessing/af3_monomer/annotate_designability_csv.py \
    --designability_stats_csv /scratch/users/rshuai/datasets/af3_pdb_monomer/designability_info/designability_stats.csv \
    --af3_dir /scratch/users/rshuai/datasets/af3_pdb_monomer \
    --out_dir /scratch/users/rshuai/datasets/af3_pdb_monomer/pdb_key_annotations