#!/bin/bash
#SBATCH --job-name=calc_metrics_denovoval_l150_250
#SBATCH -p possu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/calculate_af3_metrics_only_denovoval_len150_250_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/calculate_af3_metrics_only_denovoval_len150_250_%j.err

# ==============================================================================
# Recalculate AF3 metrics only (no GPU needed)
# Dataset: denovo_val_cifs, lengths [150,250]
# Model/step: exp37_cfg11 @ step 22500
# ==============================================================================

BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
JOB_OUT_DIR=/scratch/users/zhkim216/job_output/eval_ligand_seq_des

DATA_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp37_cfg11_denovoval_seq2_af3_len150_250/step_22500_epoch_14

mkdir -p "${JOB_OUT_DIR}" "${HYDRA_DIR}"

cd "${BASE_DIR}"

export PYTHONWARNINGS="ignore"

echo "Starting metrics recalculation: data_dir=${DATA_DIR}"

python3 allatom_design/eval/sampling/calculate_af3_metrics_only.py \
    --config-path ${BASE_DIR}/allatom_design/configs/eval/sampling \
    --config-name calculate_af3_metrics_only \
    debug=false \
    data_dir=${DATA_DIR} \
    output_csv_suffix=_recalculate \
    input_sample_is_designed=true \
    hydra.run.dir=${HYDRA_DIR}

echo "Metrics recalculation complete: data_dir=${DATA_DIR}"
