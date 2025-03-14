#!/usr/bin/env bash
conda activate allatom_design
cd /home/rshuai/research/huang_lab/allatom-design
export HF_HOME=/media/scratch/huang_lab/allatom_design/cache/huggingface
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
export CUDA_VISIBLE_DEVICES=0

python allatom_design/eval/eval_sample_sc.py \
  sampled_pdb_dir=/home/rshuai/research/huang_lab/other/proteina/outputs/lora_L32_256_8samples \
  out_dir=/media/scratch/huang_lab/allatom_design/allatom_design/proteina_lora_L32_256_8samples_sc_eval \
  exp_name=proteina_lora_L32_256_8samples_sc_eval \
  hydra.run.dir=/media/scratch/huang_lab/allatom_design/hydra_outputs