#!/bin/bash
#SBATCH --job-name=lc_seq_des_multi_2awk_gfp_sweep
#SBATCH -p possu,bioe
#SBATCH --array=0-2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/%x_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/%x_%A_%a.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
INPUTS_DIR=/scratch/users/zhkim216/datasets/GFP/260421_2AWK

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${BASE_OUT_DIR}"

cd "${BASE_DIR}"

# Ignore warnings
export PYTHONWARNINGS="ignore"

# Wandb settings
WANDB_ID=zhkim216
PROJECT=eval_ligand_seq_des

# Model / checkpoint (exp35_cfg2 step 82500)
MODEL_NAME="exp35_cfg2"
STEP=82500

# Variant sweep (array 0->noconstraint, 1->motif, 2->allpocket)
VARIANTS=(noconstraint motif allpocket)
POS_CONSTRAINT_FILES=(null pos_constraint_motif.csv pos_constraint_allpocket.csv)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
POS_CONSTRAINT_FILE=${POS_CONSTRAINT_FILES[$SLURM_ARRAY_TASK_ID]}

if [[ "${POS_CONSTRAINT_FILE}" == "null" ]]; then
    POS_CONSTRAINT_CSV=null
else
    POS_CONSTRAINT_CSV=${INPUTS_DIR}/${POS_CONSTRAINT_FILE}
fi

# Input paths
PDB_DIR=${INPUTS_DIR}/cifs
SAMPLING_INPUTS_CSV=${INPUTS_DIR}/sampling_inputs_2AWK.csv
PDB_NAME_LIST=null
VALSET_NAME="2awk_${VARIANT}"

# Denoiser train directory
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

# Eval settings
BATCH_SIZE=1
NUM_SEQS_PER_PDB=64

# Experiment name
EXP_NAME=eval_${MODEL_NAME}_${VALSET_NAME}_seq${NUM_SEQS_PER_PDB}_af3

echo "Starting evaluation for model=${MODEL_NAME}, valset=${VALSET_NAME}, step=${STEP}, pos_constraint=${POS_CONSTRAINT_CSV}"

# Automatically detect flash attention implementation
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)

if [[ "$MAJOR" -ge 8 ]]; then
    FLASH_ATTN_IMPL="triton"
else
    FLASH_ATTN_IMPL="xla"
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"

# Run evaluation script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/lc_seq_des_multi.py \
    --config-path ${BASE_DIR}/allatom_design/configs/eval/sampling \
    --config-name lc_seq_des_multi \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    debug=false \
    seed=0 \
    num_workers=${SLURM_CPUS_PER_TASK} \
    protein_only=false \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    seq_des_cfg.denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    seq_des_cfg.ckpt_cfg.start_step=${STEP} \
    seq_des_cfg.ckpt_cfg.end_step=${STEP} \
    seq_des_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    seq_des_cfg.ckpt_cfg.eval_last_ckpt=false \
    seq_des_cfg.ckpt_cfg.use_ema=true \
    seq_des_cfg.atom_mpnn.overrides.batch_size=${BATCH_SIZE} \
    seq_des_cfg.atom_mpnn.overrides.num_seqs_per_pdb=${NUM_SEQS_PER_PDB} \
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
    pos_constraint_csv=${POS_CONSTRAINT_CSV} \
    redesign_cfg.use_native_seq=false \
    pocket_cfg.pocket_distance_for_docking_metrics=8.0 \
    input_sample_is_all_atom=true \
    input_sample_is_designed=false \
    load_from_cache=false \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    pdb_cfg.sample_indices=null \
    pdb_cfg.array_id=null \
    pdb_cfg.num_arrays=null \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.evaluate_docking_consistency=false \
    struct_pred_cfg.calculate_metrics_only=false \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for model=${MODEL_NAME}, valset=${VALSET_NAME}, step=${STEP}"
