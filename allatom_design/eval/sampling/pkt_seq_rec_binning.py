"""
Pocket sequence recovery with disjoint distance binning.

Iterates checkpoints (via redesign_with_lcaliby) and writes
seq_recovery_metrics{csv_suffix}.csv per checkpoint with both cumulative
pocket_recovery_ratio_{d} columns AND new pocket_recovery_bin_{lo}_to_{hi}
columns. No AF3 / structure prediction.
"""
import gc
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import yaml

from allatom_design.eval.eval_utils.eval_setup_utils import wandb_setup
from allatom_design.eval.eval_utils.sd_data_utils import prepare_sample_dict
from allatom_design.eval.eval_utils.seq_des_utils import redesign_with_lcaliby


@hydra.main(config_path="../../configs/eval/sampling",
            config_name="pkt_seq_rec_binning", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if cfg.debug:
        cfg.wandb.project = f"debug_{cfg.wandb.project}"
        cfg.exp_name = f"debug_{cfg.exp_name}"

    log_dir = Path(wandb_setup(
        base_out_dir=cfg.base_out_dir, exp_name=cfg.exp_name,
        cfg_dict=cfg_dict, **cfg.wandb))

    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    sampling_inputs_df = (
        pd.read_csv(cfg.sampling_inputs_csv) if cfg.sampling_inputs_csv else None
    )
    pos_constraint_df = (
        pd.read_csv(cfg.pos_constraint_csv) if cfg.pos_constraint_csv else None
    )

    print("\n" + "=" * 80)
    print("Phase 1: Preparing samples")
    print("=" * 80 + "\n")
    sample_dict = prepare_sample_dict(cfg=cfg, sampling_inputs_df=sampling_inputs_df)

    array_id = cfg.pdb_cfg.get("array_id", None)
    csv_suffix = f"_array_{array_id}" if array_id is not None else ""

    bins_raw = cfg.pocket_cfg.get("pocket_distance_bins", None)
    pocket_distance_bins = (
        [tuple(b) for b in bins_raw] if bins_raw is not None else None
    )

    ckpt_steps_raw = cfg.get("ckpt_steps", None)
    ckpt_steps = list(ckpt_steps_raw) if ckpt_steps_raw else [None]

    print("\n" + "=" * 80)
    print(f"Phase 2: Redesigning pocket sequence (binned recovery) — {len(ckpt_steps)} ckpt step(s)")
    print("=" * 80 + "\n")

    for step in ckpt_steps:
        if step is not None:
            cfg.seq_des_cfg.ckpt_cfg.start_step = step
            cfg.seq_des_cfg.ckpt_cfg.end_step = step
            print(f"\n>>> Running ckpt step {step}")

        ckpt_iter = redesign_with_lcaliby(
            seed=cfg.seed,
            input_sample_is_designed=cfg.input_sample_is_designed,
            sample_dict=sample_dict,
            seq_des_cfg=cfg.seq_des_cfg,
            cif_parse_cfg=cfg.cif_cfg.parse.native,
            preprocess_cfg=cfg.preprocess_cfg.native,
            featurizer_cfg=cfg.featurizer_cfg.design,
            cif_save_cfg=cfg.cif_cfg.save,
            sampling_inputs_df=sampling_inputs_df,
            log_dir=log_dir,
            pos_constraint_df=pos_constraint_df,
            protein_only=cfg.get("protein_only", False),
            pocket_only=False,
            pocket_featurizer_cfg=None,
            pocket_distances_for_seq_recovery=cfg.pocket_cfg.pocket_distances_for_seq_recovery,
            pocket_distance_bins=pocket_distance_bins,
            csv_suffix=csv_suffix,
            guidance_cfg=None,
        )
        for sample_dict_per_ckpt, log_dir_per_ckpt, ckpt_info in ckpt_iter:
            print(f"  → step {ckpt_info['global_step']} epoch {ckpt_info['epoch']} done. CSV at {log_dir_per_ckpt}")
            del sample_dict_per_ckpt
            gc.collect()

    print("\n" + "=" * 80)
    print(f"All checkpoints done. Results: {log_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
