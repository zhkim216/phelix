import glob
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import torch
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.eval_utils import eval_metrics
from allatom_design.eval.eval_utils.eval_setup_utils import process_pdb_files
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model
from allatom_design.eval.eval_utils.seq_des_utils import (
    get_seq_des_model, run_seq_des_multistate)


@hydra.main(config_path="../../configs/eval/sampling", config_name="seq_des_single_multistate", version_base="1.3.2")
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

    # Load in PDB file to eval on
    pdb_paths = glob.glob(f"{cfg.pdb_dir}/*.pdb") + glob.glob(f"{cfg.pdb_dir}/*.cif")
    pdb_paths = natsorted(pdb_paths)[:cfg.max_num_conformers]  # subset to max_num_conformers
    processed_struct_files = process_pdb_files(pdb_paths, processed_struct_dir=f"{out_dir}/processed_structures", **cfg.pdb_processing_cfg, keep_order=True)
    conformer_struct_files = [(Path(cfg.pdb_dir).name, processed_struct_files)]

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    # # Load structure prediction model for self-consistency evaluation
    if cfg.run_self_consistency_eval:
        pred_out_dir = f"{out_dir}/preds"  # directory for structure predictions
        Path(pred_out_dir).mkdir(parents=True, exist_ok=True)
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    conformer_pdb_keys = [Path(struct_file).stem for struct_file in conformer_struct_files[0][1]]
    pos_constraint_df = pd.DataFrame({
        "pdb_key": conformer_pdb_keys,
        "fixed_pos_seq": [cfg.fixed_pos_seq] * len(conformer_pdb_keys),
        "fixed_pos_scn": [cfg.fixed_pos_scn] * len(conformer_pdb_keys),
        "fixed_pos_override_seq": [cfg.fixed_pos_override_seq] * len(conformer_pdb_keys),
        "pos_restrict_aatype": [cfg.pos_restrict_aatype] * len(conformer_pdb_keys)
    })

    # Run sequence design model
    _, aux = run_seq_des_multistate(seq_des_model["model"], seq_des_model["data_cfg"], seq_des_model["sampling_cfg"],
                                    conformer_struct_files=conformer_struct_files, device=device, pos_constraint_df=pos_constraint_df,
                                    out_dir=out_dir)

    if cfg.run_self_consistency_eval:
        id_to_metrics = eval_metrics.run_self_consistency_eval_boltz(
            aux["out_pdbs"],
            struct_pred_model,
            cfg.pdb_processing_cfg,
            out_dir=pred_out_dir)

        # Save metrics as CSV
        metrics_df = pd.DataFrame([{"record_id": rid, **m} for rid, m in id_to_metrics.items()])

        # Add additional metrics
        record_ids = [Path(x).stem.lower() for x in aux["out_pdbs"]]
        aux_df = pd.DataFrame({
            "record_id": record_ids,
            "n_conformers": aux["n_conformers"], # add n_conformers to metrics, since sometimes we are missing some conformers due to processing errors
            "U": aux["U"] # add energies to metrics
        })
        metrics_df = pd.merge(metrics_df, aux_df, on="record_id", how="left")

        metrics_df.to_csv(f"{out_dir}/self_consistency_metrics.csv", index=False)


if __name__ == "__main__":
    main()
