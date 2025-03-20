#!/bin/bash
#SBATCH --job-name=parse_sc_dataset_csv
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
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh

conda activate allatom_design

BASE_DIR=/home/users/rshuai/code/allatom-design
cd ${BASE_DIR}
python3 allatom_design/data/preprocessing/af3_monomer/parse_sc_dataset_csv.py \
    --train_sc_csv /scratch/users/rshuai/allatom_design/allatom_design/c2_aasd2/construct_sc_dataset/af3_pdb_monomer_L32_256_nl0.0/self_consistency_metrics.csv \
    --eval_sc_csv /scratch/users/rshuai/allatom_design/allatom_design/c2_aasd2/construct_sc_dataset/af3_pdb_monomer_L32_512_nl0.0_eval/self_consistency_metrics.csv \
    --struct_preds_dir /scratch/users/rshuai/datasets/af3_pdb_monomer/esmfold_preds \
    --out_dir /scratch/users/rshuai/datasets/af3_pdb_monomer/designability_info