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
from allatom_design.eval.benchmarking.benchmarking_utils import thread_sequence_onto_example


@hydra.main(config_path="../../configs/eval/benchmarking", config_name="chroma_sc_eval", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating self-consistency of Chroma designs.
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

    # Thread Chroma outputs onto PDB files, replacing undesigned residues with "X"
    threaded_pdb_dir = f"{log_dir}/threaded_pdbs"
    Path(threaded_pdb_dir).mkdir(parents=True, exist_ok=True)
    threaded_pdbs = []
    U = []  # energies

    for processed_struct_file in tqdm(processed_struct_files, desc="Threading Chroma outputs onto PDB files"):
        record_id = Path(processed_struct_file).stem

        # Read in processed structure file into features
        data_cfg = struct_pred_model["data_cfg"]
        example, input_structure = get_sd_batch([processed_struct_file], device="cpu", data_cfg=data_cfg, parallel_pool=None)

        # Load in Chroma outputs
        try:
            chroma_features = np.load(f"{cfg.chroma_outputs_dir}/features/{record_id}.npz")
        except FileNotFoundError:
            print(f"Chroma outputs not found for {record_id}, skipping...")
            continue

        chain_labels = torch.from_numpy(chroma_features["C"])  # [b, n]
        B, N = chain_labels.shape
        label_seq_id = torch.arange(N).unsqueeze(0).expand(B, N) + 1  # chroma denotes missing residues with negative values
        seq_token_ids = torch.tensor([[const.token_ids[const.prot_letter_to_token[res]] for res in seq] for seq in chroma_features["seq"]])
        chroma_restype = F.one_hot(seq_token_ids, num_classes=example["res_type"].shape[-1])

        for bi in range(B):
            if (cfg.max_samples_per_pdb is not None) and (bi == cfg.max_samples_per_pdb):
                # in case we don't want to run on all samples
                break

            example = thread_sequence_onto_example(example, chroma_restype[bi], label_seq_id[bi],
                                                   mask=chain_labels[bi] > 0)  # set missing residues to X; chroma does not update seq for missing residues but also does not use them for sampling

            # Save structure with Chroma sequence threaded on
            threaded_pdb = f"{threaded_pdb_dir}/{record_id}_sample{bi}.cif"
            write_sd_feats_to_mmcif(example, input_structure, [threaded_pdb])
            threaded_pdbs.append(threaded_pdb)

            # Get energies
            if "U" in chroma_features:
                U.append(chroma_features["U"][bi])
            else:
                U.append(np.nan)

    # Run self-consistency evaluation
    out_metrics = defaultdict(list)

    id_to_metrics = eval_metrics.run_self_consistency_eval_boltz(
        threaded_pdbs,
        struct_pred_model,
        cfg.pdb_processing_cfg,
        out_dir=log_dir)

    metrics_df = pd.DataFrame([{"record_id": rid, **m} for rid, m in id_to_metrics.items()])

    # add energies
    energies_df = pd.DataFrame({"record_id": [Path(pdb).stem for pdb in threaded_pdbs], "U": U})
    metrics_df = pd.merge(metrics_df, energies_df, on="record_id", how="left")

    # Save metrics as CSV
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
