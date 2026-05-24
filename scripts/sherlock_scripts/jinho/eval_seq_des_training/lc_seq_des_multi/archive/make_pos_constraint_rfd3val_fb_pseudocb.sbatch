#!/bin/bash
#SBATCH --job-name=mk_pos_cst_fb
#SBATCH -p possu,bioe
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/mk_pos_cst_fb_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/mk_pos_cst_fb_%A_%a.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des

mkdir -p /scratch/users/zhkim216/job_output "${HYDRA_DIR}"

cd "${BASE_DIR}"

export PYTHONWARNINGS="ignore"

# Model names and best steps (same as lc_seq_des_multi_rfd3val_fullbackbone_pseudocb.sbatch)
MODEL_NAMES=("exp34_cfg0" "exp28_cfg0" "exp28_cfg1" "exp28_cfg2" "exp28_cfg3")
BEST_STEPS=(92500 92500 92500 92500 92500)

MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
STEP=${BEST_STEPS[$SLURM_ARRAY_TASK_ID]}

VALSET_NAME="rfd3val"
EXP_NAME=eval_${MODEL_NAME}_fb_pseudocb_${VALSET_NAME}_af3

# Find samples directory dynamically (epoch number varies)
STEP_DIR=$(ls -d ${BASE_OUT_DIR}/${EXP_NAME}/step_${STEP}_epoch_*/samples 2>/dev/null | head -n 1)

if [ -z "${STEP_DIR}" ]; then
    echo "ERROR: No samples directory found for ${EXP_NAME}/step_${STEP}_epoch_*/samples"
    exit 1
fi

echo "Processing samples in: ${STEP_DIR}"
echo "Model=${MODEL_NAME}, step=${STEP}"

python3 allatom_design/eval/sampling/make_pos_constraint_df.py \
    --config-path ${BASE_DIR}/allatom_design/configs/eval/sampling \
    --config-name make_pos_constraint_df \
    debug=false \
    cif_dir=${STEP_DIR} \
    output_dir=${STEP_DIR} \
    output_prefix="" \
    pdb_list_file=null \
    metadata_path=null \
    source_is_designed=true \
    pocket_distance=8.0 \
    use_pseudocb_for_pocket_annotation=true \
    constraint_type=pocket \
    save_ligand_mpnn_csv=false \
    hydra.run.dir=${HYDRA_DIR}

echo "Done for ${EXP_NAME}"
