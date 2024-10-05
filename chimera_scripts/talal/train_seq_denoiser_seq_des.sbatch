#!/bin/bash
#SBATCH --job-name=mlm_gt_atom
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/talal/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/talal/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/talal/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
BASE_DIR=/home/talal/allatom-design
HYDRA_DIR=/scratch/hielab/talal/hydra_outputs
OUT_DIR=/scratch/hielab/talal/allatom_design
WANDB_ID=talal-widatalla
cd $BASE_DIR

EXP_NAME=mlm_gt_atom
GROUP=chimera_a8

# Dataset paths
PDB_PATH=/scratch/hielab/talal/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/hielab/talal/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_seq_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    no_wandb=false \
    data.fixed_size=256 \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    data.overwrite_cache=false \
    data.subset_length_range=[16,9999] \
    train.batch_size=32 \
    train.compile_model=false \
    model.task=seq_des \
    interpolant=mar_interpolant \
    interpolant.training_noise_schedule=uniform_squared_t \
    denoiser=minimpnn_denoiser \
    denoiser.minimpnn.model_type=graph_transformer \
    denoiser.minimpnn.augment_eps=0.0 \
    denoiser.minimpnn.graph_transformer.layers_per_nn=4 \
    denoiser.minimpnn.aggregation=sum \
    denoiser.minimpnn.graph_transformer.dim_key=3 \
    denoiser.minimpnn.graph_transformer.hidden_dim=32 \
    denoiser.minimpnn.graph_transformer.nns=[8,16,32,64] \
    denoiser.minimpnn.graph_transformer.attn_bias=true \
    denoiser.minimpnn.graph_transformer.num_heads=2 \
    denoiser.minimpnn.graph_transformer.pos_enc=true \
    optim.optimizer=noam \
    checkpointing.save_latest_every_n_epochs=20 \
    trainer.check_val_every_n_epoch=5 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
