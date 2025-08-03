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
from allatom_design.eval.eval_utils.eval_setup_utils import (get_pdb_files,
                                                             process_pdb_files,
                                                             wandb_setup)
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model
from allatom_design.eval.eval_utils.seq_des_utils import (get_seq_des_model,
                                                          run_seq_des)
import random


@hydra.main(config_path="../configs/eval", config_name="eval_seq_des_context_sweep", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating sequence design as a function of amount of random sequence context.
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

    # Load in PDB file to eval on
    pdb_files = get_pdb_files(**cfg.input_cfg)
    processed_struct_files = process_pdb_files(pdb_files, processed_struct_dir=f"{log_dir}/processed_structures", **cfg.pdb_processing_cfg)
    processed_struct_files = natsorted(processed_struct_files)

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

    # Load in fixed position sweep csv and randomize order of fixed positions
    fixed_pos_sweep_df = pd.read_csv(cfg.fixed_pos_sweep_csv)
    rng = np.random.RandomState(cfg.seed)  # fix seed explicitly just in case
    fixed_pos_sweep_df["fixed_pos_seq"] = fixed_pos_sweep_df["fixed_pos_seq"].apply(lambda x: x.split(","))
    fixed_pos_sweep_df["fixed_pos_seq"].apply(lambda x: rng.shuffle(x))
    fixed_pos_sweep_df["fixed_pos_seq"] = fixed_pos_sweep_df["fixed_pos_seq"].apply(lambda x: ",".join(x))

    # Ensure that all processed_struct_files are in fixed_pos_sweep_df
    pdb_keys = [Path(x).stem.lower() for x in processed_struct_files]
    assert set(pdb_keys).issubset(set(fixed_pos_sweep_df["pdb_key"])), f"Could not find {set(pdb_keys) - set(fixed_pos_sweep_df['pdb_key'])} in fixed_pos_sweep_df"

    # Run sequence design model for each % partial sequence context
    for t in cfg.t_sweep:
        log_dir_t = f"{log_dir}/t{t}"
        Path(log_dir_t).mkdir(parents=True, exist_ok=True)

        # We set the seed each time we run the model
        L.seed_everything(cfg.seed)

        # subset to partial sequence context
        pos_constraint_df = fixed_pos_sweep_df.copy()
        pos_constraint_df["fixed_pos_seq"] = pos_constraint_df["fixed_pos_seq"].apply(lambda x: x.split(",")[:int(t * len(x.split(",")))])
        pos_constraint_df["fixed_pos_seq"] = pos_constraint_df["fixed_pos_seq"].apply(lambda x: ",".join(x))
        if cfg.fix_scn:
            # fix ground truth sidechains as well
            pos_constraint_df["fixed_pos_scn"] = pos_constraint_df["fixed_pos_seq"]

        # save pos_constraint_df to csv
        pos_constraint_df.to_csv(f"{log_dir_t}/pos_constraint_df.csv", index=False)

        # Run sequence design with this partial context
        outputs = run_seq_des(seq_des_model["model"], seq_des_model["data_cfg"], seq_des_model["sampling_cfg"],
                              struct_file_paths=processed_struct_files, device=device, pos_constraint_df=pos_constraint_df,
                              out_dir=log_dir_t)

        # Save outputs to CSV
        record_ids = [Path(x).stem.lower() for x in outputs["out_pdbs"]]
        output_df = pd.DataFrame({"record_id": record_ids, "pdb_key": outputs["pdb_keys"], "seq": outputs["seqs"], "input_seq": outputs["input_seqs"]})
        output_df.to_csv(f"{log_dir_t}/seq_des_outputs.csv", index=False)

        if cfg.run_self_consistency_eval:
            id_to_metrics = eval_metrics.run_self_consistency_eval_boltz(
                outputs["out_pdbs"],
                struct_pred_model,
                cfg.pdb_processing_cfg,
                out_dir=f"{log_dir_t}/preds")

            # Save metrics as CSV
            metrics_df = pd.DataFrame([{"record_id": rid, **m} for rid, m in id_to_metrics.items()])
            metrics_df.to_csv(f"{log_dir_t}/self_consistency_metrics.csv", index=False)

            if not cfg.wandb.no_wandb:
                # Aggregate results
                sc_metrics = defaultdict(list)
                for record_id, metrics in id_to_metrics.items():
                    for k, v in metrics.items():
                        sc_metrics[f"{k}"].append(v)

                # Update metrics
                out_metrics = {f"seq_des/mean/{k}": np.nanmean(v) for k, v in sc_metrics.items() if k != "record_id"}
                out_metrics.update({f"seq_des/median/{k}": np.nanmedian(v) for k, v in sc_metrics.items() if k != "record_id"})

                # Log metrics to wandb
                out_metrics["t_seq"] = t
                wandb.log(out_metrics)


if __name__ == "__main__":
    main()
