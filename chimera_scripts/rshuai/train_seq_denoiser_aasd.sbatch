#!/bin/bash
#SBATCH --job-name=c2_aasd2
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
OUT_DIR=/scratch/hielab/rshuai/allatom_design
WANDB_ID=richardshuai
cd $BASE_DIR

EXP_NAME=c2_aasd2
GROUP=chimera_c2

# Dataset paths
PDB_PATH=/scratch/hielab/talal/datasets/af3_pdb
DESIGNABILITY_CSV=null

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_seq_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    no_wandb=false \
    data.fixed_size=1024 \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    data.subset_length_range=[16,9999] \
    train.batch_size=16 \
    trainer.accumulate_grad_batches=2 \
    train.compile_model=true \
    model.task=allatom_seq_des \
    interpolant.training_noise_schedule=uniform_sqrt_t \
    denoiser=minimpnn_denoiser \
    denoiser.minimpnn.augment_eps=0.3 \
    denoiser.minimpnn.per_residue_eps=true \
    denoiser.minimpnn.use_gvp=false \
    denoiser.minimpnn.n_layers=5 \
    denoiser.minimpnn.n_channel=128 \
    denoiser.confidence_module.enabled=true \
    optim.optimizer=noam \
    checkpointing.save_latest_every_n_steps=20000 \
    checkpointing.save_ema_every_n_steps=5000 \
    trainer.check_val_every_n_epoch=5 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
