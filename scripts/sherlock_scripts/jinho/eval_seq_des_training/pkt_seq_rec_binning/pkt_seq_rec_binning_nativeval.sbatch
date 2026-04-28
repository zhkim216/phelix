#!/bin/bash
#SBATCH --job-name=pkt_seq_rec_binning
#SBATCH -p possu,bioe
#SBATCH --array=0-9
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/%x_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/%x_%A_%a.err

# 10 tasks: 5 models * 2 ligand modes. Each task = ONE python invocation
# that loops internally over 4 ckpt steps (defined in yaml ckpt_steps),
# so the python process owns its own GPU memory lifecycle across steps.
# Pocket sequence recovery on nativeval, binned by distance to ligand:
#   (0, 4]  (4, 6]  (6, 8]  (8, 10]   plus cumulative cutoffs at 4/6/8/10.

BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/pkt_seq_rec_binning

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${BASE_OUT_DIR}"
cd "${BASE_DIR}"
export PYTHONWARNINGS="ignore"

WANDB_ID=zhkim216
PROJECT=eval_ligand_seq_des

MODEL_NAMES=("exp37_cfg2" "exp37_cfg5" "exp36_cfg1" "exp37_cfg8" "exp37_cfg11")  # 5
PROTEIN_ONLY_VALS=(false true)                                                    # 2

MODE_IDX=$((SLURM_ARRAY_TASK_ID % 2))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 2))

MODEL_NAME=${MODEL_NAMES[$MODEL_IDX]}
PROTEIN_ONLY=${PROTEIN_ONLY_VALS[$MODE_IDX]}

if [[ "${PROTEIN_ONLY}" == "true" ]]; then MODE_TAG="protonly"; else MODE_TAG="withlig"; fi

PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/native_val_cifs/cifs
SAMPLING_INPUTS_CSV=/scratch/users/zhkim216/datasets/val_cifs/native_val_cifs/sampling_inputs_nativeval_sm_selected.csv
PDB_NAME_LIST=/scratch/users/zhkim216/datasets/val_cifs/native_val_cifs/nativeval_sm_selected.txt
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

EXP_NAME=pkt_bin_${MODEL_NAME}_${MODE_TAG}

echo "=========================================================="
echo "TASK ${SLURM_ARRAY_TASK_ID}"
echo "  model=${MODEL_NAME}   mode=${MODE_TAG}"
echo "  exp_name=${EXP_NAME}   ckpt_steps from yaml"
echo "=========================================================="

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/pkt_seq_rec_binning.py \
    --config-path ${BASE_DIR}/allatom_design/configs/eval/sampling \
    --config-name pkt_seq_rec_binning \
    wandb.no_wandb=true \
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
    seq_des_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    seq_des_cfg.ckpt_cfg.eval_last_ckpt=false \
    seq_des_cfg.ckpt_cfg.use_ema=true \
    seq_des_cfg.atom_mpnn.overrides.batch_size=8 \
    seq_des_cfg.atom_mpnn.overrides.num_seqs_per_pdb=4 \
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
    pos_constraint_csv=null \
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
    hydra.run.dir=${HYDRA_DIR}

echo "DONE: ${EXP_NAME}"
