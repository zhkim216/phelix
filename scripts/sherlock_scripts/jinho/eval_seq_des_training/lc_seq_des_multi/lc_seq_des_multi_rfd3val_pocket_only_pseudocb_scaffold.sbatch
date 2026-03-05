#!/bin/bash
#SBATCH --job-name=lc_seq_des_multi_po_scaffold
#SBATCH -p possu,bioe,owners
#SBATCH --array=0-799
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_po_scaffold_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_po_scaffold_%A_%a.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des

mkdir -p /scratch/users/zhkim216/job_output "${HYDRA_DIR}" "${BASE_OUT_DIR}"

cd "${BASE_DIR}"

# Ignore warnings
export PYTHONWARNINGS="ignore"

# Wandb settings
WANDB_ID=zhkim216
PROJECT=eval_ligand_seq_des

# 2D array indexing: CONFIG_ID selects model config, DATA_SHARD_ID splits data by subfolder
NUM_CONFIGS=16
NUM_DATA_SHARDS=50
CONFIG_ID=$((SLURM_ARRAY_TASK_ID / NUM_DATA_SHARDS))
DATA_SHARD_ID=$((SLURM_ARRAY_TASK_ID % NUM_DATA_SHARDS))

# Model names, crop distances, and best steps (by pocket 5A seq recovery, lmpnnval pseudocb)
MODEL_NAMES=(    "exp30_cfg0" "exp30_cfg0" "exp30_cfg0" "exp30_cfg0"
                 "exp30_cfg1" "exp30_cfg1" "exp30_cfg1" "exp30_cfg1"
                 "exp30_cfg2" "exp30_cfg2" "exp30_cfg2" "exp30_cfg2"
                 "exp33_cfg0" "exp33_cfg0" "exp33_cfg0" "exp33_cfg0")
POCKET_CROP_DISTANCES=(6.0  8.0  10.0  12.0
                       6.0  8.0  10.0  12.0
                       6.0  8.0  10.0  12.0
                       6.0  8.0  10.0  12.0)
BEST_STEPS=(52500 52500 12500  2500
            72500 62500 82500 82500
            42500 42500 32500 72500
            52500 12500 92500 72500)

MODEL_NAME=${MODEL_NAMES[$CONFIG_ID]}
POCKET_CROP_DISTANCE=${POCKET_CROP_DISTANCES[$CONFIG_ID]}
STEP=${BEST_STEPS[$CONFIG_ID]}

VALSET_NAME="rfd3val"
EXP_NAME=eval_${MODEL_NAME}_po_pseudocb_cd${POCKET_CROP_DISTANCE}A_${VALSET_NAME}_af3
EXP2_NAME=eval_${MODEL_NAME}_po_pseudocb_cd${POCKET_CROP_DISTANCE}A_${VALSET_NAME}_lcaliby_af3

# Resolve step directory (epoch number varies, use glob safely)
STEP_DIR=$(ls -d ${BASE_OUT_DIR}/${EXP_NAME}/step_${STEP}_epoch_* 2>/dev/null | head -n 1)

if [ -z "${STEP_DIR}" ]; then
    echo "ERROR: No step directory found for ${EXP_NAME}/step_${STEP}_epoch_*"
    exit 1
fi

# Input paths
PDB_DIR=${STEP_DIR}/samples_pocket_only_full
SAMPLING_INPUTS_CSV=null
PDB_NAME_LIST=null
POS_CONSTRAINT_CSV=${STEP_DIR}/samples/_pos_constraint_pocket_8.0A.csv

# Denoiser train directory
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/exp18_cfg3 # L-Caliby (C-alpha)

# Experiment name 
echo "Starting designing scaffold sequences using L-Caliby, with designed sequences by model=${MODEL_NAME}, pocket_crop_distance=${POCKET_CROP_DISTANCE}A, step=${STEP}, valset=${VALSET_NAME}"
echo "Config ${CONFIG_ID}/${NUM_CONFIGS}, data shard ${DATA_SHARD_ID}/${NUM_DATA_SHARDS}"

# Automatically detect flash attention implementation
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)

if [[ "$MAJOR" -ge 8 ]]; then
    FLASH_ATTN_IMPL="triton"
else
    FLASH_ATTN_IMPL="xla"
fi
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"

# Run evaluation script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/lc_seq_des_multi.py \
    --config-path ${BASE_DIR}/allatom_design/configs/eval/sampling \
    --config-name lc_seq_des_multi \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=baseline \
    debug=false \
    seed=0 \
    num_workers=${SLURM_CPUS_PER_TASK} \
    protein_only=false \
    exp_name=${EXP2_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    seq_des_cfg.denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    seq_des_cfg.ckpt_cfg.start_step=82500 \
    seq_des_cfg.ckpt_cfg.end_step=82500 \
    seq_des_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    seq_des_cfg.ckpt_cfg.eval_last_ckpt=false \
    seq_des_cfg.ckpt_cfg.use_ema=true \
    seq_des_cfg.atom_mpnn.overrides.batch_size=16 \
    seq_des_cfg.atom_mpnn.overrides.num_seqs_per_pdb=1 \
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
    pos_constraint_csv=${POS_CONSTRAINT_CSV} \
    redesign_cfg.use_native_seq=false \
    input_sample_is_all_atom=true \
    input_sample_is_designed=true \
    load_from_cache=false \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=null \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=true \
    pdb_cfg.split_by_subfolder=true \
    pdb_cfg.array_id=${DATA_SHARD_ID} \
    pdb_cfg.num_arrays=${NUM_DATA_SHARDS} \
    'pdb_cfg.sample_indices=[0,1,2,3,4]' \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    struct_pred_cfg.evaluate_docking_consistency=false \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for model=${MODEL_NAME}, data shard=${DATA_SHARD_ID}"
