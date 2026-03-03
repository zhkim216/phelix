#!/bin/bash
#SBATCH --job-name=lc_seq_des_multi
#SBATCH -p possu,bioe,owners
#SBATCH --array=0-4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_%A_%a.err

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

# ===== Parallelization settings =====
NUM_DATA_CHUNKS=5  # Number of data-parallel chunks per combination

COMBINATIONS=(    
    "exp18_cfg3,72500,true"
)
# "exp19_cfg0,2500,false"    
# "exp19_cfg0,12500,false"    
# ...

# 2D indexing: combo x data_chunk
NUM_COMBOS=${#COMBINATIONS[@]}
COMBO_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_DATA_CHUNKS ))
DATA_CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_DATA_CHUNKS ))

# Parse the current combination
IFS=',' read -r MODEL_NAME STEP PROTEIN_ONLY <<< "${COMBINATIONS[$COMBO_IDX]}"

# Input paths
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/lmpnn_val_cifs/cifs
SAMPLING_INPUTS_CSV=/scratch/users/zhkim216/datasets/val_cifs/lmpnn_val_cifs/sampling_inputs_lmpnnval_small_molecule_filtered.csv
POS_CONSTRAINT_CSV=null
PDB_NAME_LIST=/scratch/users/zhkim216/datasets/val_cifs/lmpnn_val_cifs/lmpnnval_small_molecule_filtered_nogaphetero.txt
VALSET_NAME="lmpnnval_sm_filt_nogaphetero"

# Denoiser train directory
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

# Experiment name 
if [ "${PROTEIN_ONLY}" = "true" ]; then
    EXP_NAME=eval_${MODEL_NAME}_${VALSET_NAME}_af3_protonly
else
    EXP_NAME=eval_${MODEL_NAME}_${VALSET_NAME}_af3
fi

echo "Starting evaluation for model=${MODEL_NAME}, step=${STEP}, protein_only=${PROTEIN_ONLY}, valset=${VALSET_NAME}, data_chunk=${DATA_CHUNK_IDX}/${NUM_DATA_CHUNKS}"

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
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=baseline \
    debug=false \
    seed=0 \
    num_workers=${SLURM_CPUS_PER_TASK} \
    protein_only=${PROTEIN_ONLY} \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    seq_des_cfg.denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    seq_des_cfg.ckpt_cfg.start_step=${STEP} \
    seq_des_cfg.ckpt_cfg.end_step=${STEP} \
    seq_des_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    seq_des_cfg.ckpt_cfg.eval_last_ckpt=false \
    seq_des_cfg.ckpt_cfg.use_ema=true \
    seq_des_cfg.atom_mpnn.overrides.batch_size=8 \
    seq_des_cfg.atom_mpnn.overrides.num_seqs_per_pdb=1 \
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
    pos_constraint_csv=${POS_CONSTRAINT_CSV} \
    redesign_cfg.use_native_seq=false \
    pocket_cfg.pocket_distance_for_metrics=6.0 \
    input_sample_is_all_atom=true \
    input_sample_is_designed=false \
    load_from_cache=false \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    pdb_cfg.sample_indices=null \
    pdb_cfg.array_id=${DATA_CHUNK_IDX} \
    pdb_cfg.num_arrays=${NUM_DATA_CHUNKS} \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for model=${MODEL_NAME}, data_chunk=${DATA_CHUNK_IDX}"