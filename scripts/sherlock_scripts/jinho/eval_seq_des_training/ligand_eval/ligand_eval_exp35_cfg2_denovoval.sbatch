#!/bin/bash
#SBATCH --job-name=le_exp35_cfg2
#SBATCH -p possu,owners
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/le_exp35_cfg2_denovoval_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/le_exp35_cfg2_denovoval_%A_%a.err

# ==============================================================================
# Ligand evaluation (PB → Glide → PB) for exp35_cfg2 on denovo_val designable
# Array job: each task processes a chunk of selected samples
# Within each task: multiprocessing with NUM_WORKERS parallel workers
# ==============================================================================

# --- Configurable parameters ---
NUM_WORKERS=4
LIGAND_RMSD_CUTOFF=2.0
LIGAND_PLDDT_CUTOFF=80
# --------------------------------

# Directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des

AF3_EVAL_DIR=${BASE_OUT_DIR}/eval_exp35_cfg2_denovoval_full_af3/step_82500_epoch_52
OUTPUT_DIR=${BASE_OUT_DIR}/eval_exp35_cfg2_denovoval_full_ligand_eval_new/step_82500_epoch_52
SAMPLE_DIR=${AF3_EVAL_DIR}/samples
SCHRODINGER_PATH=/scratch/users/zhkim216/software/schrodinger2025-3

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${OUTPUT_DIR}"

cd "${BASE_DIR}"
export PYTHONWARNINGS="ignore"

echo "Starting ligand eval for exp35_cfg2 denovoval (array_task=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}, workers=${NUM_WORKERS})"

python3 -m allatom_design.eval.glide.run_ligand_eval_batch \
    --config-path ${BASE_DIR}/allatom_design/configs/eval/glide \
    --config-name run_ligand_eval_batch \
    af3_eval_dir=${AF3_EVAL_DIR} \
    output_dir=${OUTPUT_DIR} \
    sample_dir=${SAMPLE_DIR} \
    num_workers=${NUM_WORKERS} \
    debug=false \
    selection.ligand_rmsd_cutoff=${LIGAND_RMSD_CUTOFF} \
    selection.ligand_plddt_cutoff=${LIGAND_PLDDT_CUTOFF} \
    schrodinger.schrodinger_path=${SCHRODINGER_PATH} \
    hydra.run.dir=${HYDRA_DIR}

echo "Ligand eval complete for exp35_cfg2 denovoval (array_task=${SLURM_ARRAY_TASK_ID})"
