from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval import eval_metrics
from allatom_design.eval.fampnn_utils import get_seq_des_model, run_fampnn
from allatom_design.eval.folding_utils import get_struct_pred_model


@hydra.main(config_path="../../configs/eval/sampling", config_name="fampnn_single", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for designing sequences for a single PDB.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Make output directory
    out_dir = cfg.out_dir  # base output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)  # create output directory

    # Preserve config
    with open(Path(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    # Load structure prediction model for self-consistency evaluation
    if cfg.run_self_consistency_eval:
        pred_out_dir = f"{out_dir}/preds"  # directory for structure predictions
        Path(pred_out_dir).mkdir(parents=True, exist_ok=True)
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    # Create single sample fixed pos df
    pdb_name = Path(cfg.pdb_path).name
    pos_constraint_df = pd.DataFrame({
        "pdb_name": [pdb_name],
        "fixed_pos_seq": [cfg.fixed_pos_seq],
        "fixed_pos_scn": [cfg.fixed_pos_scn],
        "fixed_pos_override_seq": [cfg.fixed_pos_override_seq],
        "pos_restrict_aatype": [cfg.pos_restrict_aatype]
    })

    # Run FAMPNN
    _, aux = run_fampnn(seq_des_model["fampnn_model"], seq_des_model["fampnn_cfg"],
                        pdb_paths=[cfg.pdb_path], device=device, pos_constraint_df=pos_constraint_df,
                        out_dir=cfg.out_dir)

    if cfg.run_self_consistency_eval:
        codes_sc_info = eval_metrics.run_self_consistency_eval(
            aux["out_pdbs"],
            None,  # no MPNN model to use sequence from PDB
            struct_pred_model,
            device,
            out_dir=pred_out_dir,
            temp_dir=f"{pred_out_dir}/tmp"
        )

        # Aggregate results
        codes_metrics = defaultdict(list)
        for pdb in aux["out_pdbs"]:
            for k, v in codes_sc_info[pdb]["sc_metrics"].items():
                codes_metrics[f"codes_{k}"].append(v.item())

        out_df = pd.DataFrame(codes_metrics)
        out_df.to_csv(f"{out_dir}/self_consistency_metrics.csv", index=False)


if __name__ == "__main__":
    main()
