#!/usr/bin/env bash
ENV_DIR=/media/scratch/envs
source ${ENV_DIR}/allatom_design/bin/activate

cd /home/rshuai/research/huang_lab/allatom-design
export HF_HOME=/media/scratch/huang_lab/allatom_design/cache/huggingface

# Loop over partial_t_scn values
for t in $(seq 0.1 0.1 1.0); do
  echo "Running sidechain_pack with partial_t_scn=${t}..."
  CUDA_VISIBLE_DEVICES=0  \
  python /home/rshuai/research/huang_lab/allatom-design/allatom_design/eval/sampling/sidechain_pack.py \
    sd_ckpt=/media/scratch/huang_lab/allatom_design/allatom_design/c0_aasd4/post_hoc_ema_ckpts/ema-step300000-std0.250.ckpt \
    exp_name=casp13_14_15_tscn_${t} \
    batch_size=8 \
    data.pdb_path=/media/scratch/datasets/casp13_14_15 \
    scn_diffusion.noise_schedule.c=1.5 \
    scn_diffusion.num_steps=50 \
    num_pdbs=null \
    partial_t_scn=${t} \
    seed=1 \
    subset_length_range=[0,9999] \
    num_samples_per_pdb=1
done
