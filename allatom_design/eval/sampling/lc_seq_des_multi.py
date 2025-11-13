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


@hydra.main(config_path="../../configs_local/eval/sampling", config_name="lc_seq_des_multi", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for designing sequences for multiple PDBs.
    - Single checkpoint: 기존 동작과 동일하게 한 번 실행하고 단일 CSV 저장
    - Sweep mode: 지정된 디렉토리의 여러 체크포인트에 대해 반복 실행, 체크포인트별 CSV 저장 및 (옵션) W&B 로깅
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

    # # # Load structure prediction model for self-consistency evaluation
    # if cfg.run_self_consistency_eval:
    #     pred_out_dir = f"{log_dir}/preds"  # directory for structure predictions
    #     Path(pred_out_dir).mkdir(parents=True, exist_ok=True)
    #     struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

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

        # Save outputs to CSV
        output_df = pd.DataFrame(outputs)
        output_df.to_csv(f"{log_dir}/seq_des_outputs.csv", index=False)
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

            # Save outputs to CSV with checkpoint-specific name
            output_df = pd.DataFrame(outputs)
            if step_val is not None and epoch_val is not None:
                out_csv = Path(log_dir, f"seq_des_outputs_step{step_val}-epoch{epoch_val}.csv")
            else:
                out_csv = Path(log_dir, f"seq_des_outputs_{ckpt_path.stem}.csv")
            output_df.to_csv(out_csv, index=False)

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

    # if cfg.run_self_consistency_eval:
    #     id_to_metrics = eval_metrics.run_self_consistency_eval_boltz(
    #         outputs["out_pdbs"],
    #         struct_pred_model,
    #         cfg.pdb_processing_cfg,
    #         out_dir=pred_out_dir)

    #     # Save metrics as CSV
    #     metrics_df = pd.DataFrame([{"record_id": rid, **m} for rid, m in id_to_metrics.items()])
    #     metrics_df.to_csv(f"{log_dir}/self_consistency_metrics.csv", index=False)

    #     if not cfg.wandb.no_wandb:
    #         # Aggregate results
    #         sc_metrics = defaultdict(list)
    #         for record_id, metrics in id_to_metrics.items():
    #             for k, v in metrics.items():
    #                 sc_metrics[f"{k}"].append(v)

    #         # Update metrics
    #         out_metrics = {f"seq_des/mean/{k}": np.nanmean(v) for k, v in sc_metrics.items() if k != "record_id"}
    #         out_metrics.update({f"seq_des/median/{k}": np.nanmedian(v) for k, v in sc_metrics.items() if k != "record_id"})

    #         # Log metrics to wandb
    #         wandb.log(out_metrics, step=0)


if __name__ == "__main__":
    main()
    
