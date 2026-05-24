#!/bin/bash
#SBATCH --job-name=lc_seq_des_multi_po
#SBATCH -p possu,bioe,owners
#SBATCH --array=0-24
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_po_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_po_%A_%a.err

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

# Model names and steps
MODEL_NAMES=("exp30_cfg0" "exp30_cfg1" "exp30_cfg2" "exp33_cfg0" "exp33_cfg1")
POCKET_CROP_DISTANCES=(6.0 8.0 10.0 12.0 14.0)

MODEL_IDX=$((SLURM_ARRAY_TASK_ID % 5))
POCKET_CROP_DISTANCE_IDX=$((SLURM_ARRAY_TASK_ID / 5))

MODEL_NAME=${MODEL_NAMES[$MODEL_IDX]}
POCKET_CROP_DISTANCE=${POCKET_CROP_DISTANCES[$POCKET_CROP_DISTANCE_IDX]}

# Input paths
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/lmpnn_val_cifs/cifs
SAMPLING_INPUTS_CSV=/scratch/users/zhkim216/datasets/val_cifs/lmpnn_val_cifs/sampling_inputs_lmpnnval_small_molecule_filtered.csv
PDB_NAME_LIST=/scratch/users/zhkim216/datasets/val_cifs/lmpnn_val_cifs/lmpnnval_small_molecule_filtered.txt
VALSET_NAME="lmpnnval_sm_filt_full" #"lmpnnval_sm_filt_nogaphetero" or "lmpnnval_sm_filt_full"

POS_CONSTRAINT_CSV=null

# Denoiser train directory
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

# Experiment name
EXP_NAME=eval_${MODEL_NAME}_po_cd${POCKET_CROP_DISTANCE}A_${VALSET_NAME}_af3

echo "Starting evaluation for model=${MODEL_NAME} (idx=${MODEL_IDX}), pocket_crop_distance=${POCKET_CROP_DISTANCE}A (idx=${POCKET_CROP_DISTANCE_IDX}), valset=${VALSET_NAME}"

# Run evaluation script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/lc_seq_des_multi.py \
    --config-path ${BASE_DIR}/allatom_design/configs/eval/sampling \
    --config-name lc_seq_des_pocket_only_multi \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=baseline \
    debug=false \
    seed=0 \
    num_workers=${SLURM_CPUS_PER_TASK} \
    protein_only=false \
    pocket_only=true \
    pocket_featurizer_cfg.pocket_crop_distance=${POCKET_CROP_DISTANCE} \
    pocket_featurizer_cfg.spatial_crop_radius=30.0 \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    seq_des_cfg.denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    seq_des_cfg.ckpt_cfg.start_step=2500 \
    seq_des_cfg.ckpt_cfg.end_step=92500 \
    seq_des_cfg.ckpt_cfg.eval_every_n_ckpts=4 \
    seq_des_cfg.ckpt_cfg.eval_last_ckpt=false \
    seq_des_cfg.ckpt_cfg.use_ema=true \
    seq_des_cfg.atom_mpnn.overrides.batch_size=8 \
    seq_des_cfg.atom_mpnn.overrides.num_seqs_per_pdb=4 \
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
    pos_constraint_csv=${POS_CONSTRAINT_CSV} \
    redesign_cfg.use_native_seq=false \
    input_sample_is_all_atom=true \
    input_sample_is_designed=false \
    load_from_cache=false \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=".cif" \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=false \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for model=${MODEL_NAME}"
