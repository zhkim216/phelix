#!/bin/bash
#SBATCH --job-name=aasd_pdb_0.3
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
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

EXP_NAME=aasd_pdb_0.3
GROUP=chimera_a8

# Dataset paths
PDB_PATH=/scratch/hielab/talal/datasets/af3_pdb
DESIGNABILITY_CSV=/scratch/hielab/talal/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_seq_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    cuda=true \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    no_wandb=false \
    data.fixed_size=256 \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=false \
    data.overwrite_cache=false \
    train.batch_size=32 \
    train.compile_model=true \
    trainer.max_epochs=100 \
    model.task=allatom_seq_des \
    interpolant=mar_interpolant \
    interpolant.training_noise_schedule=uniform_squared_t \
    denoiser=minimpnn_denoiser \
    denoiser.minimpnn.model_type=baseline \
    denoiser.minimpnn.use_residue_transformer=false \
    denoiser.minimpnn.use_gvp=false \
    denoiser.minimpnn.augment_eps=0.3 \
    denoiser.minimpnn.return_embedding=last \
    denoiser.minimpnn.gvp.rotate_out=false \
    denoiser.minimpnn.residue_transformer.condition_on_seq=false \
    denoiser.minimpnn.residue_transformer.edge_attn_bias=true \
    denoiser.scn_diffusion_module.training_batch_size_mult=4 \
    optim.optimizer=noam \
    checkpointing.save_latest_every_n_steps=50000 \
    trainer.check_val_every_n_epoch=1 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}