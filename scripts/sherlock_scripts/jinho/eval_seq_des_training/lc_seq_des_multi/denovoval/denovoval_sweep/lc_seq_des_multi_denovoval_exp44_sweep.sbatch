#!/bin/bash
#SBATCH --job-name=lc_seq_des_denovoval_sweep_exp44
#SBATCH -p possu,bioe
#SBATCH --array=0-21
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_denovoval_sweep_exp44_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_denovoval_sweep_exp44_%A_%a.err

MODEL_STEP_COMBOS=(
    "exp44_cfg1:62500"
    "exp44_cfg1:102500"
    "exp44_cfg1:142500"
    "exp44_cfg2:62500"
    "exp44_cfg3:62500"
    "exp44_cfg4:62500"
    "exp44_cfg4:102500"
    "exp44_cfg4:142500"
    "exp44_cfg5:62500"
    "exp44_cfg6:62500"
    "exp44_cfg6:102500"
)
NUM_CHUNKS=2

# Array index decoding: SLURM_ARRAY_TASK_ID -> (COMBO_IDX, CHUNK_IDX)
# Total tasks = NUM_COMBOS * NUM_CHUNKS (= 13 * 5 = 65 -> --array=0-64)
NUM_COMBOS=${#MODEL_STEP_COMBOS[@]}
COMBO_IDX=$((SLURM_ARRAY_TASK_ID / NUM_CHUNKS))
CHUNK_IDX=$((SLURM_ARRAY_TASK_ID % NUM_CHUNKS))

COMBO=${MODEL_STEP_COMBOS[$COMBO_IDX]}
MODEL_NAME=${COMBO%%:*}
STEP=${COMBO##*:}

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${BASE_OUT_DIR}"

cd "${BASE_DIR}"

# Ignore warnings
export PYTHONWARNINGS="ignore"

# Wandb settings
WANDB_ID=zhkim216
PROJECT=eval_ligand_seq_des

# Input paths (denovo val CIFs, flat directory)
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/denovo_val_cifs
VALSET_NAME="denovoval_len150250"

# Denoiser train directory
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

# Experiment name
EXP_NAME=eval_${MODEL_NAME}_${VALSET_NAME}_af3

echo "Starting evaluation for model=${MODEL_NAME}, step=${STEP}, valset=${VALSET_NAME}, chunk=${CHUNK_IDX}/${NUM_CHUNKS} (array_task=${SLURM_ARRAY_TASK_ID})"

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
    wandb.group=${VALSET_NAME} \
    debug=false \
    seed=0 \
    num_workers=${SLURM_CPUS_PER_TASK} \
    protein_only=false \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    run_seq_des=true \
    guidance_cfg.enabled=false \
    seq_des_cfg.denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    seq_des_cfg.ckpt_cfg.start_step=${STEP} \
    seq_des_cfg.ckpt_cfg.end_step=${STEP} \
    seq_des_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    seq_des_cfg.ckpt_cfg.eval_last_ckpt=false \
    seq_des_cfg.ckpt_cfg.use_ema=true \
    seq_des_cfg.atom_mpnn.overrides.batch_size=2 \
    seq_des_cfg.atom_mpnn.overrides.num_seqs_per_pdb=1 \
    seq_des_cfg.atom_mpnn.overrides.use_potts_sampling=true \
    seq_des_cfg.atom_mpnn.overrides.omit_aas=null \
    seq_des_cfg.atom_mpnn.overrides.noise_labels=null \
    seq_des_cfg.atom_mpnn.overrides.potts_sampling_cfg.potts_sweeps=500 \
    seq_des_cfg.atom_mpnn.overrides.potts_sampling_cfg.lcp_expand_edge_idx_fix=true \
    seq_des_cfg.atom_mpnn.overrides.potts_sampling_cfg.rejection_step=false \
    sampling_inputs_csv=null \
    pos_constraint_csv=null \
    redesign_cfg.use_native_seq=false \
    redesign_cfg.native_cif_dir=null \
    pocket_cfg.pocket_distance_for_docking_metrics=10.0 \
    input_sample_is_all_atom=true \
    input_sample_is_designed=true \
    load_from_cache=false \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=null \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    pdb_cfg.array_id=${CHUNK_IDX} \
    pdb_cfg.num_arrays=${NUM_CHUNKS} \
    'pdb_cfg.sample_indices=[0,1,2,3,4]' \
    'pdb_cfg.sample_lengths=[150,250]' \
    featurizer_cfg.design.is_inference=true \
    featurizer_cfg.design.training_structure_noise=0.0 \
    featurizer_cfg.design.apply_random_augmentation=true \
    featurizer_cfg.design.pocket_distance=6.0 \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.evaluate_docking_consistency=false \
    struct_pred_cfg.calculate_metrics_only=false \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for model=${MODEL_NAME}, step=${STEP}, chunk=${CHUNK_IDX}"
