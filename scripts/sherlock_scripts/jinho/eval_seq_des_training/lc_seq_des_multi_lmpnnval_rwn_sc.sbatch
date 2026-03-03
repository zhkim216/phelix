#!/bin/bash
#SBATCH --job-name=lc_seq_des_multi_rwn_sc
#SBATCH -p possu,bioe,owners
#SBATCH --array=0-29
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_rwn_sc_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_rwn_sc_%A_%a.err

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
PROJECT=ligand_seq_des

# Parallelization settings
NUM_ARRAYS=10  

# Define combinations of models & steps
COMBINATIONS=(
    "exp18_cfg3,72500,false,4.0"        
    "exp18_cfg3,72500,false,5.0"  
    "exp18_cfg3,72500,false,6.0"  
)

# --- 수정: SLURM_ARRAY_TASK_ID에서 combination 인덱스와 로컬 array ID 계산 ---
COMBO_IDX=$((SLURM_ARRAY_TASK_ID / NUM_ARRAYS))
LOCAL_ARRAY_ID=$((SLURM_ARRAY_TASK_ID % NUM_ARRAYS))

# Parse the current combination
IFS=',' read -r MODEL_NAME STEP PROTEIN_ONLY REDESIGN_POCKET_DISTANCE <<< "${COMBINATIONS[$COMBO_IDX]}"

# Input paths
PDB_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp18_cfg3_lmpnnval_sm_filt_nogaphetero_af3/step_72500_epoch_46/samples
SAMPLING_INPUTS_CSV=null
PDB_NAME_LIST=null
VALSET_NAME="lmpnnval_sm_filt_nogaphetero"
POS_CONSTRAINT_CSV=null
NATIVE_CIF_DIR=/scratch/users/zhkim216/datasets/val_cifs/lmpnn_val_cifs/cifs

# Denoiser train directory
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

# Experiment name 
EXP_NAME=eval_${MODEL_NAME}_step_${STEP}_rwn_${REDESIGN_POCKET_DISTANCE}_${VALSET_NAME}_af3

echo "Starting evaluation for model=${MODEL_NAME}, step=${STEP}, protein_only=${PROTEIN_ONLY}, valset=${VALSET_NAME}"
echo "Combination ${COMBO_IDX}, Array task ${LOCAL_ARRAY_ID} of ${NUM_ARRAYS} (SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID})"

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
    redesign_cfg.use_native_seq=true \
    redesign_cfg.pocket_distance=${REDESIGN_POCKET_DISTANCE} \
    redesign_cfg.native_cif_dir=${NATIVE_CIF_DIR} \
    pocket_cfg.pocket_distance_for_metrics=6.0 \
    input_sample_is_all_atom=false \
    input_sample_is_designed=true \
    load_from_cache=false \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    pdb_cfg.sample_indices=null \
    pdb_cfg.array_id=${LOCAL_ARRAY_ID} \
    pdb_cfg.num_arrays=${NUM_ARRAYS} \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.calculate_metrics_only=false \
    struct_pred_cfg.af3.runner_path=/home/users/zhkim216/code/allatom-design/alphafold3/run_alphafold_debug_sherlock.py \
    struct_pred_cfg.af3.inference_config.base.model_dir=/scratch/users/zhkim216/model_params/af3 \
    struct_pred_cfg.af3.inference_config.base.db_dir=/scratch/users/zhkim216/af3_databases \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=triton \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for model=${MODEL_NAME}, array_task=${SLURM_ARRAY_TASK_ID}"