#!/bin/bash
#SBATCH --job-name=get_af3_pdb_keys
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
python3 allatom_design/data/preprocessing/get_af3_monomer_pdb_key_list.py \
    --chain_clustering_csv /scratch/hielab/talal/datasets/af3_pdb/data_caches/data_caches/val_clusterings/protein_chain_cluster_mapping.csv \
    --phase eval \
    --out_dir $SCRATCH/datasets/af3_pdb_monomer