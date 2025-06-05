import glob
import itertools
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.eval_utils import eval_metrics
from allatom_design.eval.eval_utils.eval_setup_utils import (process_pdb_files,
                                                             wandb_setup)
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model
from allatom_design.eval.eval_utils.seq_des_utils import (
    get_seq_des_model, run_seq_des_multistate)


@hydra.main(config_path="../../configs/eval/sampling", config_name="seq_des_multi_multistate", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for designing sequences for multiple conformers of multiple PDBs.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up wandb logging / output directory
    log_dir = wandb_setup(base_out_dir=cfg.base_out_dir, exp_name=cfg.exp_name, cfg_dict=cfg_dict, **cfg.wandb)

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load in PDB files to eval on
    pdb_names = [Path(x).name for x in glob.glob(f"{cfg.conformer_dir}/*")]

    # first, collect per-PDB conformer lists
    conformer_groups = []
    for pdb_name in pdb_names:
        all_conformers = natsorted(glob.glob(f"{cfg.conformer_dir}/{pdb_name}/*"))
        primary_conformer = f"{cfg.conformer_dir}/{pdb_name}/{pdb_name}.cif"
        all_conformers.remove(primary_conformer)
        if cfg.include_primary_conformer:
            conformers = [primary_conformer] + all_conformers[:cfg.max_num_conformers - 1]
        else:
            conformers = all_conformers[:cfg.max_num_conformers]
        conformer_groups.append((pdb_name, conformers))

    # flatten and process everything in one go
    all_confs_flat = [c for _, group in conformer_groups for c in group]
    processed_flat = process_pdb_files(all_confs_flat, processed_struct_dir=f"{log_dir}/processed_structures", **cfg.pdb_processing_cfg, keep_order=True)

    # then split the flat list back into per-PDB results
    conformer_struct_files = []
    offset = 0
    for pdb_name, group in conformer_groups:
        n = len(group)
        conformer_struct_files.append((pdb_name, processed_flat[offset:offset + n]))
        offset += n

    # filter out conformers that failed to process
    conformer_struct_files = [(pdb_name, [x for x in struct_files if x is not None]) for pdb_name, struct_files in conformer_struct_files]

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    # # Load structure prediction model for self-consistency evaluation
    if cfg.run_self_consistency_eval:
        pred_out_dir = f"{log_dir}/preds"  # directory for structure predictions
        Path(pred_out_dir).mkdir(parents=True, exist_ok=True)
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    # Run sequence design model
    _, aux = run_seq_des_multistate(seq_des_model["model"], seq_des_model["data_cfg"], seq_des_model["sampling_cfg"],
                                    conformer_struct_files=conformer_struct_files, device=device, pos_constraint_df=None,
                                    out_dir=log_dir)

    if cfg.run_self_consistency_eval:
        id_to_metrics = eval_metrics.run_self_consistency_eval_boltz(
            aux["out_pdbs"],
            struct_pred_model,
            cfg.pdb_processing_cfg,
            out_dir=pred_out_dir)

        # Save metrics as CSV
        metrics_df = pd.DataFrame([{"record_id": rid, **m} for rid, m in id_to_metrics.items()])

        # Add n_conformers to metrics, since sometimes we are missing some conformers due to processing errors
        record_ids = [Path(x).stem for x in aux["out_pdbs"]]
        n_conformers_df = pd.DataFrame({"record_id": record_ids, "n_conformers": aux["n_conformers"]})
        metrics_df = pd.merge(metrics_df, n_conformers_df, on="record_id", how="left")

        metrics_df.to_csv(f"{log_dir}/self_consistency_metrics.csv", index=False)

        if not cfg.wandb.no_wandb:
            # Aggregate results
            sc_metrics = defaultdict(list)
            for record_id, metrics in id_to_metrics.items():
                for k, v in metrics.items():
                    sc_metrics[f"{k}"].append(v)

            # Update metrics
            out_metrics = {f"seq_des/mean/{k}": np.nanmean(v) for k, v in sc_metrics.items() if k not in ["record_id", "n_conformers"]}
            out_metrics.update({f"seq_des/median/{k}": np.nanmedian(v) for k, v in sc_metrics.items() if k not in ["record_id", "n_conformers"]})

            # Log metrics to wandb
            wandb.log(out_metrics, step=0)


if __name__ == "__main__":
    main()
