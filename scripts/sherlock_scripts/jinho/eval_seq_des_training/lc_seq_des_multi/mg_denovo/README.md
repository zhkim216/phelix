# Mg denovo RASA N100 lc_seq_des_multi on Sherlock

This directory contains the Sherlock handoff scripts for evaluating the Mg RASA N100 subset with `lc_seq_des_multi`.

## Inputs

Expected Sherlock dataset root:

```bash
/scratch/users/zhkim216/datasets/val_cifs/mg_denovo_val_cifs_test
```

If copying the raw generated CIFs again, do not quote the glob:

```bash
cp /scratch/users/zhkim216/rfd3_mg_binding_len150_1000/*.cif.gz \
  /scratch/users/zhkim216/datasets/val_cifs/mg_denovo_val_cifs_test/
```

The `sampling_inputs_*.csv` does not need `sample_path`. For `lc_seq_des_multi`, structure discovery comes from `pdb_cfg.pdb_dir` and `pdb_cfg.pdb_name_list`; the CSV is only used for optional metadata such as `pdb_id` and `query_pn_unit_iids`.

## Prepare Files

Run this once on Sherlock:

```bash
cd /home/users/zhkim216/code/elix
python3 scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/mg_denovo/prepare_mg_denovo_rasa_n100_sherlock.py --dry-run
python3 scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/mg_denovo/prepare_mg_denovo_rasa_n100_sherlock.py --force
```

Generated files:

```text
/scratch/users/zhkim216/datasets/val_cifs/mg_denovo_val_cifs_test/cifs/*.cif
/scratch/users/zhkim216/datasets/val_cifs/mg_denovo_val_cifs_test/mg_denovo_rasa_le0p25_uniform_bins_N100.txt
/scratch/users/zhkim216/datasets/val_cifs/mg_denovo_val_cifs_test/mg_denovo_rasa_le0p25_uniform_bins_N100_smoke2.txt
/scratch/users/zhkim216/datasets/val_cifs/mg_denovo_val_cifs_test/sampling_inputs_mg_denovo_rasa_le0p25_uniform_bins_N100.csv
/scratch/users/zhkim216/datasets/val_cifs/mg_denovo_val_cifs_test/mg_denovo_rasa_le0p25_uniform_bins_N100_manifest.json
```

## Submit

Smoke run:

```bash
PROJECT_ROOT=/home/users/zhkim216/code/elix MODE=smoke \
  bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
  scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/mg_denovo/lc_seq_des_multi_mg_denovo_rasa_n100_mg_proto_gpt.sbatch
```

Single-checkpoint full RASA N100 run:

```bash
PROJECT_ROOT=/home/users/zhkim216/code/elix \
  bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
  scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/mg_denovo/lc_seq_des_multi_mg_denovo_rasa_n100_mg_proto_gpt.sbatch
```

Multi-model, per-step sweep:

```bash
# All 6 models x 3 steps: 18 tasks
PROJECT_ROOT=/home/users/zhkim216/code/elix \
  bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
  scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/mg_denovo/lc_seq_des_multi_mg_denovo_rasa_n100_per_step_sweep.sbatch

# One step across all 6 models
PROJECT_ROOT=/home/users/zhkim216/code/elix STEP_FILTER=22500 \
  bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
  scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/mg_denovo/lc_seq_des_multi_mg_denovo_rasa_n100_per_step_sweep.sbatch

# One model and one step
PROJECT_ROOT=/home/users/zhkim216/code/elix MODEL_FILTER=mg_proto_gpt STEP_FILTER=42500 \
  bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
  scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/mg_denovo/lc_seq_des_multi_mg_denovo_rasa_n100_per_step_sweep.sbatch

# Smoke version of the full sweep
PROJECT_ROOT=/home/users/zhkim216/code/elix MODE=smoke \
  bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
  scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/mg_denovo/lc_seq_des_multi_mg_denovo_rasa_n100_per_step_sweep.sbatch
```

The Elix wrapper does not take `sbatch --array` or `sbatch --export` passthrough arguments. The per-step sweep keeps `#SBATCH --array=0-17`; when `MODEL_FILTER` or `STEP_FILTER` is set, tasks outside the selected combinations exit without work.

The per-step sweep covers these models:

```text
mg_proto_no_filter
mg_proto_len512_no_filter
mg_proto_substring
mg_proto_len512_substring
mg_proto_gpt
mg_proto_len512_gpt
```

and these steps:

```text
2500
22500
42500
```

Full mode and smoke mode both use `NUM_CHUNKS=1`. Full mode uses AF3 self-consistency with `num_recycles=10` and `num_diffusion_samples=5`. Smoke mode uses `1/1`.

Main output root:

```text
/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/mg_denovo_rasa_le0p25_uniform_bins_N100_${MODEL_NAME}_ema0p99_no_guidance_af3/step_${STEP}_epoch_*
```

The runner sets `pocket_cfg.pocket_distance_for_docking_metrics=5.0`, `seq_des_cfg.ckpt_cfg.use_ema=true`, `guidance_cfg.enabled=false`, and `struct_pred_cfg.evaluate_docking_consistency=false`.
