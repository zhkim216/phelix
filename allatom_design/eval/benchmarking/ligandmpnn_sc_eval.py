from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import allatom_design.data.const as const
from allatom_design.data.write.mmcif import write_sd_feats_to_mmcif
from allatom_design.eval.eval_utils import eval_metrics
from allatom_design.eval.eval_utils.eval_setup_utils import (get_pdb_files,
                                                             process_pdb_files,
                                                             wandb_setup)
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model
from allatom_design.eval.eval_utils.seq_des_utils import get_sd_batch


@hydra.main(config_path="../../configs/eval/benchmarking", config_name="ligandmpnn_sc_eval", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating self-consistency of LigandMPNN designs.
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

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in PDB files to eval on
    pdb_files = get_pdb_files(**cfg.input_cfg)
    processed_struct_files = process_pdb_files(pdb_files, processed_struct_dir=f"{log_dir}/processed_structures", **cfg.pdb_processing_cfg)

    # Load in structure prediction model for self-consistency evals
    struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    # Thread ligandMPNN outputs onto PDB files, replacing undesigned residues with "X"
    threaded_pdb_dir = f"{log_dir}/threaded_pdbs"
    Path(threaded_pdb_dir).mkdir(parents=True, exist_ok=True)
    threaded_pdbs = []

    for processed_struct_file in tqdm(processed_struct_files, desc="Threading ligandMPNN outputs onto PDB files"):
        record_id = Path(processed_struct_file).stem

        # Read in processed structure file into features
        data_cfg = struct_pred_model["data_cfg"]
        example, input_structure = get_sd_batch([processed_struct_file], device="cpu", data_cfg=data_cfg, parallel_pool=None)

        # For now, make sure there is only one protein chain
        protein_mask = example["mol_type"] == const.chain_type_ids["PROTEIN"]
        n_prot_chains = len(example["asym_id"][protein_mask].unique())
        if n_prot_chains > 1:
            raise ValueError(f"Found {n_prot_chains} protein chains in {processed_struct_file}. For now, we only support threading sequences onto single-chain proteins.")

        # Load in ligandMPNN outputs
        ligandmpnn_features = np.load(f"{cfg.ligandmpnn_outputs_dir}/features/{record_id}.npz")
        label_seq_id = torch.from_numpy(ligandmpnn_features["R_idx_original"])  # 1-indexed
        seq_token_ids = torch.tensor([[const.token_ids[const.prot_letter_to_token[res]] for res in seq] for seq in ligandmpnn_features["seq"]])
        ligandmpnn_restype = F.one_hot(seq_token_ids, num_classes=example["res_type"].shape[-1])
        B = ligandmpnn_restype.shape[0]

        for bi in range(B):
            # Set all protein residues to X and erase all sidechain coordinates
            protein_res_type = torch.full_like(protein_mask, const.token_ids["UNK"], dtype=torch.long)
            protein_res_type = F.one_hot(protein_res_type, num_classes=example["res_type"].shape[-1])  # [1, n_protein, 33]
            protein_res_type = protein_res_type.squeeze(0)  # temporarily squeeze out batch dimension
            protein_res_type[label_seq_id[bi] - 1] = ligandmpnn_restype[bi]  # label_seq_id is 1-indexed
            example["res_type"][protein_mask] = protein_res_type.unsqueeze(0)
            example["coords"][example["prot_scn_atom_mask"].bool()] = 0.0

            # Save structure with ligandMPNN sequence threaded on
            threaded_pdb = f"{threaded_pdb_dir}/{record_id}_sample{bi}.cif"
            write_sd_feats_to_mmcif(example, input_structure, [threaded_pdb])
            threaded_pdbs.append(threaded_pdb)

    # Run self-consistency evaluation
    out_metrics = defaultdict(list)

    id_to_metrics = eval_metrics.run_self_consistency_eval_boltz(
        threaded_pdbs,
        struct_pred_model,
        cfg.pdb_processing_cfg,
        out_dir=log_dir)

    # Save metrics as CSV
    metrics_df = pd.DataFrame([{"record_id": rid, **m} for rid, m in id_to_metrics.items()])
    metrics_df.to_csv(f"{log_dir}/sc_metrics.csv", index=False)

    # Aggregate results
    sc_metrics = defaultdict(list)
    for record_id, metrics in id_to_metrics.items():
        for k, v in metrics.items():
            sc_metrics[f"{k}"].append(v)

    # Update metrics
    out_metrics = {f"seq_des/mean/{k}": np.nanmean(v) for k, v in sc_metrics.items() if k != "record_id"}
    out_metrics.update({f"seq_des/median/{k}": np.nanmedian(v) for k, v in sc_metrics.items() if k != "record_id"})

    # Log metrics to wandb
    if not cfg.wandb.no_wandb:
        wandb.log(out_metrics, step=0)


if __name__ == "__main__":
    main()
