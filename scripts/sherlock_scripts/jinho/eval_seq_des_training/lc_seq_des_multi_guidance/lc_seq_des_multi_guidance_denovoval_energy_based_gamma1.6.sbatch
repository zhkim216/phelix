#!/bin/bash
#SBATCH --job-name=lc_seq_des_guidance_denovoval_energy_based_g1.6
#SBATCH -p possu,bioe
#SBATCH --array=0-1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_guidance_denovoval_energy_based_g1.6_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_guidance_denovoval_energy_based_g1.6_%A_%a.err

# ==============================================================================
# DLMC guidance full run on denovo_val_cifs (exp35_cfg2 step 82500).
# Single-gamma variant (energy-based, clamp OFF, pocket=10A). See GAMMA below.
#
# sample_indices=[0,1,2,3,4], sample_lengths=[150,250], num_seqs_per_pdb=1
# NUM_CHUNKS=2 (--array=0-1)
# AF3 SS: num_recycles=10, num_diffusion_samples=5
# ==============================================================================
MODEL_NAMES=("exp35_cfg2")
STEPS=(82500)
NUM_CHUNKS=2
GAMMA=1.6

NUM_SEQS_PER_PDB=1

NUM_MODELS=${#MODEL_NAMES[@]}
NUM_STEPS=${#STEPS[@]}
TASKS_PER_MODEL=$((NUM_STEPS * NUM_CHUNKS))

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_MODEL))
REMAINDER=$((SLURM_ARRAY_TASK_ID % TASKS_PER_MODEL))
STEP_IDX=$((REMAINDER / NUM_CHUNKS))
CHUNK_IDX=$((REMAINDER % NUM_CHUNKS))

MODEL_NAME=${MODEL_NAMES[$MODEL_IDX]}
STEP=${STEPS[$STEP_IDX]}

BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${BASE_OUT_DIR}"

cd "${BASE_DIR}"

export PYTHONWARNINGS="ignore"

WANDB_ID=zhkim216
PROJECT=eval_ligand_seq_des

PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/denovo_val_cifs
VALSET_NAME="denovoval_guidance_energy_based"

DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

EXP_NAME=eval_${MODEL_NAME}_${VALSET_NAME}_len150_250_gamma${GAMMA}

echo "Starting guidance run for model=${MODEL_NAME}, step=${STEP}, valset=${VALSET_NAME}, gamma=${GAMMA}, chunk=${CHUNK_IDX}/${NUM_CHUNKS} (array_task=${SLURM_ARRAY_TASK_ID})"

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)

if [[ "$MAJOR" -ge 8 ]]; then
    FLASH_ATTN_IMPL="triton"
else
    FLASH_ATTN_IMPL="xla"
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"

# NOTE: batch_size=4 (halved from the usual 8) since guidance runs cond+uncond
# forward passes and doubles Potts-decoder memory.
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/lc_seq_des_multi.py \
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
    seq_des_cfg.denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    seq_des_cfg.ckpt_cfg.start_step=${STEP} \
    seq_des_cfg.ckpt_cfg.end_step=${STEP} \
    seq_des_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    seq_des_cfg.ckpt_cfg.eval_last_ckpt=false \
    seq_des_cfg.ckpt_cfg.use_ema=true \
    seq_des_cfg.atom_mpnn.overrides.batch_size=4 \
    seq_des_cfg.atom_mpnn.overrides.num_seqs_per_pdb=${NUM_SEQS_PER_PDB} \
    sampling_inputs_csv=null \
    pos_constraint_csv=null \
    redesign_cfg.use_native_seq=false \
    redesign_cfg.native_cif_dir=null \
    pocket_cfg.pocket_distance_for_docking_metrics=8.0 \
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
    guidance_cfg.enabled=true \
    "guidance_cfg.gamma_list=[${GAMMA}]" \
    guidance_cfg.uncond_mode=protein_only \
    guidance_cfg.clamp_mix_prob=false \
    guidance_cfg.guidance_mode=energy \
    guidance_cfg.pocket_distance=10.0 \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.evaluate_docking_consistency=false \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Guidance run complete for model=${MODEL_NAME}, step=${STEP}, gamma=${GAMMA}, chunk=${CHUNK_IDX}"
