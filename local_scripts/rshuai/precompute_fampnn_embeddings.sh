#!/usr/bin/env bash
conda activate allatom_design
cd /home/rshuai/research/huang_lab/allatom-design
export HF_HOME=/media/scratch/huang_lab/allatom_design/cache/huggingface
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
export CUDA_VISIBLE_DEVICES=0

python /home/rshuai/research/huang_lab/allatom-design/allatom_design/eval/precompute_embeddings.py \
  compute_mpnn=false \
  compute_esm3=false \
  fampnn.checkpoint_path=/media/scratch/huang_lab/allatom_design/allatom_design/c2_aasd2/post_hoc_ema_ckpts/ema-step300000-std0.220.ckpt \
  train_pdb_key_file=/media/scratch/datasets/ingraham_cath_dataset/train_pdb_keys.list \
  eval_pdb_key_file=/media/scratch/datasets/ingraham_cath_dataset/eval_pdb_keys.list \
  eval2_pdb_key_file=/media/scratch/datasets/ingraham_cath_dataset/eval2_pdb_keys.list \
  pdb_name_ext='' \
  pdbs_dir=/media/scratch/datasets/ingraham_cath_dataset/pdb_store \
  out_dir=/media/scratch/datasets/ingraham_cath_dataset/fampnn_embeddings
