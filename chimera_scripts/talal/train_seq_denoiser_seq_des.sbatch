#!/bin/bash
#SBATCH --job-name=mlm_gvp_trans
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:59
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

EXP_NAME=mlm_gvp_trans
GROUP=chimera_a8

# Dataset paths
PDB_PATH=/scratch/hielab/talal/datasets/ingraham_cath_dataset

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_seq_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=8 \
    no_wandb=false \
    data.fixed_size=256 \
    data.pdb_path=${PDB_PATH} \
    data.overwrite_cache=false \
    train.batch_size=32 \
    train.compile_model=true \
    trainer.max_epochs=500 \
    model.task=seq_des \
    interpolant=mar_interpolant \
    interpolant.training_noise_schedule=uniform_squared_t \
    denoiser=fampnn_denoiser \
    denoiser.fampnn.model_type=sidechain \
    denoiser.fampnn.use_residue_transformer=true \
    denoiser.fampnn.use_gvp=true \
    denoiser.fampnn.augment_eps=0 \
    denoiser.fampnn.return_embedding=last \
    denoiser.fampnn.gvp.rotate_out=false \
    denoiser.scn_diffusion_module.training_batch_size_mult=4 \
    optim.optimizer=noam \
    checkpointing.save_latest_every_n_epochs=20 \
    trainer.check_val_every_n_epoch=5 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
