#!/bin/bash
#SBATCH --job-name=process_pdbs
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
CCD_DIR=/scratch/hielab/rshuai/datasets/boltz
CCD_FILENAME=ccd.rdb
redis-server --dir ${CCD_DIR} --dbfilename ${CCD_FILENAME} --port 7777 &
python3 allatom_design/data/preprocessing/af3_pdb_monomer_boltz/process_pdbs.py \
    af3_pdb_monomer_path=/scratch/hielab/rshuai/datasets/af3_pdb_monomer \
    out_dir=/scratch/hielab/rshuai/datasets/af3_pdb_monomer_boltz/preprocessing/process_pdbs \
    pdb_path=/scratch/hielab/rshuai/datasets/af3_pdb_monomer_boltz \
    num_workers=16