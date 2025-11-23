from collections import defaultdict
from pathlib import Path
import re

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.eval_utils import eval_metrics
from allatom_design.eval.eval_utils.eval_setup_utils import (get_pdb_files,
                                                             wandb_setup,
                                                             get_cached_example_files,
                                                             )
# from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model
from allatom_design.eval.eval_utils.seq_des_utils import (get_seq_des_model,
                                                          run_lc_seq_des)


@hydra.main(config_path="../../configs/eval/sampling", config_name="lc_seq_des_multi", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for designing sequences for multiple PDBs.
    - Single checkpoint: Run evaluation on only the single checkpoint
    - Sweep mode: Run evaluations on the checkpoints in the checkpoint directory
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up wandb logging / output directory
    sweep_enabled = hasattr(cfg, "sweep_cfg") and cfg.sweep_cfg.enabled
    wandb_kwargs = OmegaConf.to_container(cfg.wandb, resolve=True)
    if sweep_enabled and getattr(cfg.sweep_cfg, "wandb_log", False):
        wandb_kwargs["no_wandb"] = False
    exp_name = cfg.exp_name if not sweep_enabled else f"{cfg.exp_name}_sweep"
    if cfg.debug:
        exp_name = f"debug_{exp_name}"
    log_dir = wandb_setup(base_out_dir=cfg.base_out_dir, exp_name=exp_name, cfg_dict=cfg_dict, **wandb_kwargs)

    # Load in metadata
    metadata = pd.read_parquet(cfg.metadata_path)
    pdb_keys = metadata['pdb_id'].tolist()

    if cfg.debug:
        pdb_keys = pdb_keys[:cfg.num_sample_debug]

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load in PDB file to eval on
    if cfg.input_cfg.load_from_cache:
        pdb_files = get_cached_example_files(cached_example_path=cfg.input_cfg.load_cache_cfg.cached_example_path, pdb_name_list=pdb_keys, \
                                             pdb_name_ext=cfg.input_cfg.load_cache_cfg.pdb_name_ext, n_subsample=cfg.input_cfg.load_cache_cfg.n_subsample)
    else:
        pdb_files = get_pdb_files(pdb_dir=cfg.input_cfg.pdb_cfg.pdb_dir, pdb_name_list=pdb_keys, \
                              pdb_name_ext=cfg.input_cfg.pdb_cfg.pdb_name_ext, n_subsample=cfg.input_cfg.pdb_cfg.n_subsample)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model
    if not sweep_enabled:
        seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)
    
    # Read in fixed positions
    if cfg.pos_constraint_csv is not None:
        pos_constraint_df = pd.read_csv(cfg.pos_constraint_csv)
    else:
        pos_constraint_df = None

    if not sweep_enabled:
        # Run sequence design model (single checkpoint)
        outputs = run_lc_seq_des(model = seq_des_model["model"], 
                                 load_from_cache=cfg.input_cfg.load_from_cache,
                                 featurizer_cfg = cfg.featurizer_cfg, 
                                 sampling_cfg = seq_des_model["sampling_cfg"],                          
                                 metadata = metadata,
                                 pdb_paths=pdb_files, device=device, 
                                 pos_constraint_df=pos_constraint_df,
                                 out_dir=log_dir)

        # Save outputs to CSV (split per-sample vs per-PDB)
        sample_len = len(outputs["example_id"])
        sample_cols = {
            "example_id": outputs["example_id"],
            "seq": outputs["seq"],
            "U": outputs["U"],
        }
        # add per-sample metrics if lengths match
        for k in ("sample_seq_recovery", "sample_sp_seq_recovery"):
            if k in outputs and isinstance(outputs[k], list) and len(outputs[k]) == sample_len:
                sample_cols[k] = outputs[k]
        sample_df = pd.DataFrame(sample_cols)
        sample_df.to_csv(f"{log_dir}/seq_des_outputs_samples.csv", index=False)

        # per-PDB averages
        ex_ids = list(dict.fromkeys(outputs["example_id"]))
        if "sample_avg_seq_recovery" in outputs and "sample_avg_sp_seq_recovery" in outputs:
            per_pdb_df = pd.DataFrame({
                "example_id": ex_ids,
                "sample_avg_seq_recovery": outputs["sample_avg_seq_recovery"],
                "sample_avg_sp_seq_recovery": outputs["sample_avg_sp_seq_recovery"],
            })
            per_pdb_df.to_csv(f"{log_dir}/seq_des_outputs_per_pdb.csv", index=False)

        # summary (scalars)
        summary = {
            "total_avg_seq_recovery": outputs.get("total_avg_seq_recovery", None),
            "total_avg_sp_seq_recovery": outputs.get("total_avg_sp_seq_recovery", None),
        }
        with open(Path(log_dir, "summary.yaml"), "w") as f:
            yaml.safe_dump(summary, f)
    else:
        # Sweep over checkpoints in the specified directory
        ckpt_dir = Path(cfg.sweep_cfg.ckpt_dir)
        ckpt_paths = sorted(ckpt_dir.glob(cfg.sweep_cfg.ckpt_glob))
        if getattr(cfg.sweep_cfg, "max_ckpts", None) is not None:
            ckpt_paths = ckpt_paths[:cfg.sweep_cfg.max_ckpts]

        for ckpt_path in ckpt_paths:
            m = re.search(r"step(\d+)-epoch(\d+)", ckpt_path.stem)
            step_val = int(m.group(1)) if m else None
            epoch_val = int(m.group(2)) if m else None

            # Load models for each checkpoint
            seq_cfg_dict = OmegaConf.to_container(cfg.seq_des_cfg, resolve=True)
            seq_cfg = OmegaConf.create(seq_cfg_dict)
            seq_cfg.atom_mpnn.ckpt_path = str(ckpt_path)
            seq_des_model = get_seq_des_model(seq_cfg, device=device)

            outputs = run_lc_seq_des(model = seq_des_model["model"], 
                                     load_from_cache=cfg.input_cfg.load_from_cache,
                                     featurizer_cfg = cfg.featurizer_cfg, 
                                     sampling_cfg = seq_des_model["sampling_cfg"],                          
                                     metadata = metadata,
                                     pdb_paths=pdb_files, device=device, 
                                     pos_constraint_df=pos_constraint_df,
                                     out_dir=log_dir)

            # Save outputs to CSV with checkpoint-specific name (split)
            base_stem = (
                f"seq_des_outputs_step{step_val}-epoch{epoch_val}"
                if (step_val is not None and epoch_val is not None)
                else f"seq_des_outputs_{ckpt_path.stem}"
            )

            sample_len = len(outputs["example_id"])
            sample_cols = {
                "example_id": outputs["example_id"],
                "seq": outputs["seq"],
                "U": outputs["U"],
            }
            for k in ("sample_seq_recovery", "sample_sp_seq_recovery"):
                if k in outputs and isinstance(outputs[k], list) and len(outputs[k]) == sample_len:
                    sample_cols[k] = outputs[k]
            sample_df = pd.DataFrame(sample_cols)
            sample_df.to_csv(Path(log_dir, f"{base_stem}_samples.csv"), index=False)

            if "sample_avg_seq_recovery" in outputs and "sample_avg_sp_seq_recovery" in outputs:
                ex_ids = list(dict.fromkeys(outputs["example_id"]))
                per_pdb_df = pd.DataFrame({
                    "example_id": ex_ids,
                    "sample_avg_seq_recovery": outputs["sample_avg_seq_recovery"],
                    "sample_avg_sp_seq_recovery": outputs["sample_avg_sp_seq_recovery"],
                })
                per_pdb_df.to_csv(Path(log_dir, f"{base_stem}_per_pdb.csv"), index=False)

            # summary (scalars) per checkpoint
            summary = {
                "total_avg_seq_recovery": outputs.get("total_avg_seq_recovery", None),
                "total_avg_sp_seq_recovery": outputs.get("total_avg_sp_seq_recovery", None),
            }
            with open(Path(log_dir, f"{base_stem}_summary.yaml"), "w") as f:
                yaml.safe_dump(summary, f)

            # Wandb logging per checkpoint in sweep
            if not wandb_kwargs.get("no_wandb", True):
                log_step = step_val if step_val is not None else 0
                wandb.log({                    
                    "eval/total_avg_seq_acc": outputs["total_avg_seq_recovery"],
                    "eval/total_avg_sp_seq_acc": outputs["total_avg_sp_seq_recovery"],
                }, step=log_step)

            print (f"Wandb logged for checkpoint: {ckpt_path}")
            # Free memory
            del seq_des_model
            if device == "cuda":
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    
