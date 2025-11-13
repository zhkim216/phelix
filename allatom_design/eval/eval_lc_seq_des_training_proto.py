import json
import os
import re
import shutil
import subprocess
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
from tqdm import tqdm

from allatom_design.data import data as data_utils
from allatom_design.eval.eval_utils.eval_setup_utils import (
    get_pdb_files, get_training_checkpoints, process_pdb_files, wandb_setup)
from allatom_design.eval.eval_utils.seq_des_utils import (
    get_seq_des_model, run_seq_des)


def _chain_letters(n: int) -> list[str]:
    letters = []
    base = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    if n <= 26:
        return base[:n]
    # Extend like A, B, ..., Z, AA, BA, CA, ... (reverse spreadsheet style used in AF3 docs)
    letters.extend(base)
    idx = 0
    while len(letters) < n:
        letters.extend([f"{base[i]}{base[idx]}" for i in range(26)])
        idx += 1
    return letters[:n]


def _make_af3_single_json(job_name: str,
                          chain_seqs: list[str],
                          model_seeds: list[int]) -> dict:
    sequences = []
    chain_ids = _chain_letters(len(chain_seqs))
    for cid, seq in zip(chain_ids, chain_seqs):
        sequences.append({
            "protein": {
                "id": cid,
                "sequence": seq,
                "unpairedMsa": "",
                "pairedMsa": ""
            }
        })
    return {
        "name": job_name,
        "sequences": sequences,
        "modelSeeds": model_seeds,
        "dialect": "alphafold3",
        "version": 1,
    }


def _run_af3(json_path: str,
             out_dir: str,
             runner_path: str,
             extra_args: list[str] | None = None) -> None:
    cmd = [
        "python",
        runner_path,
        f"--json_path={json_path}",
        f"--output_dir={out_dir}",
        "--run_data_pipeline=True",
        "--run_inference=True",
    ]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def _find_best_pred_path(out_dir: str, job_name: str) -> str | None:
    # Prefer top-level best-ranking output written with name=job_name
    candidates = list(Path(out_dir).glob(f"{job_name}*.cif")) + list(Path(out_dir).glob(f"{job_name}*.pdb"))
    if candidates:
        # Prefer .cif if present
        cif_candidates = [p for p in candidates if p.suffix.lower() == ".cif"]
        return str(sorted(cif_candidates or candidates)[0])

    # Otherwise, look into seed-sample subdirectories for any cif/pdb
    pattern_dirs = sorted(Path(out_dir).glob("seed-*_*"))
    for d in pattern_dirs:
        files = list(d.glob("*.cif")) + list(d.glob("*.pdb"))
        if files:
            cif_files = [p for p in files if p.suffix.lower() == ".cif"]
            return str(sorted(cif_files or files)[0])
    return None


def _compute_sc_metrics(sample_cif: str,
                        pred_struct_path: str,
                        temp_dir: str) -> dict[str, float]:
    sample_feats = data_utils.load_feats_from_pdb(sample_cif)
    pred_feats = data_utils.load_feats_from_pdb(pred_struct_path)

    # Both tensors to shape [1, N, 37, 3] etc.
    coords1 = pred_feats["all_atom_positions"][None]
    coords2 = sample_feats["all_atom_positions"][None]
    # Use sample mask for metrics; shapes must match
    atom_mask = sample_feats["all_atom_mask"][None]

    metrics_to_compute = ["sc_ca_rmsd", "sc_ca_tm", "sc_aa_rmsd"]
    metrics, _ = eval_compute_structure_metrics(coords1, coords2, atom_mask, metrics_to_compute, temp_dir)
    # Convert tensor values to float
    out = {}
    for k, v in metrics.items():
        try:
            out[k] = float(v.item())
        except Exception:
            # Some metrics like sc_ca_tm may be tensors already shaped [B]
            try:
                out[k] = float(v.squeeze().item())
            except Exception:
                out[k] = np.nan
    return out


def eval_compute_structure_metrics(coords1: torch.Tensor,
                                   coords2: torch.Tensor,
                                   atom_mask: torch.Tensor,
                                   metrics_to_compute: list[str],
                                   temp_dir: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    from allatom_design.eval.eval_utils import eval_metrics as em
    return em.compute_structure_metrics(coords1, coords2, atom_mask, metrics_to_compute=metrics_to_compute, temp_dir=temp_dir)


@hydra.main(config_path="../configs_local/eval", config_name="eval_lc_seq_des_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Sequence denoiser training eval with AF3 self-consistency.

    Flow per checkpoint:
    - sample sequences on processed input structures
    - for each sample, build AF3 JSON (MSA-free) and run AF3
    - compute structure self-consistency metrics vs sampled structure
    - save CSV and optionally log to wandb
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Seeds and determinism
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Logging / output root
    log_dir = wandb_setup(base_out_dir=cfg.base_out_dir, exp_name=cfg.exp_name, cfg_dict=cfg_dict, **cfg.wandb)

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Device
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Inputs -> processed
    pdb_files = get_pdb_files(**cfg.input_cfg)
    processed_struct_files = process_pdb_files(pdb_files, processed_struct_dir=f"{log_dir}/processed_structures", **cfg.pdb_processing_cfg)
    processed_struct_files = natsorted(processed_struct_files)

    # # Denoiser checkpoints
    # sd_ckpts, pattern = get_training_checkpoints(cfg.denoiser_train_dir, "seq_denoiser",
    #                                              cfg.eval_every_n_ckpts, cfg.start_step, cfg.end_step)

    # # AF3 config defaults
    # af3_runner_path = cfg.af3.get("runner_path", "/home/possu/jinho/allatom-design/alphafold3/run_alphafold_debug_local.py")
    # af3_model_seeds = list(cfg.af3.get("model_seeds", [42]))
    # af3_extra_args = list(cfg.af3.get("extra_args", []))

    # pbar = tqdm(sd_ckpts, desc="Evaluating checkpoints (AF3 self-consistency)...")
    # for sd_ckpt in pbar:
    #     match = pattern.search(Path(sd_ckpt).name)
    #     global_step, epoch = int(match.group(1)), int(match.group(2))
    #     pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")

    #     # Per-ckpt out dir
    #     log_dir_i = f"{log_dir}/step_{global_step}_epoch_{epoch}"
    #     Path(log_dir_i).mkdir(parents=True, exist_ok=True)

    #     # Load sequence design model
    #     cfg.seq_des_cfg.atom_mpnn.ckpt_path = sd_ckpt
    #     seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    #     # Reset seed per checkpoint
    #     L.seed_everything(cfg.seed)

    #     # Run sequence design
    #     outputs = run_seq_des(seq_des_model["model"], seq_des_model["data_cfg"], seq_des_model["sampling_cfg"],
    #                           pdb_paths=processed_struct_files, device=device, out_dir=log_dir_i)
    #     sampled_cifs = outputs["out_pdbs"]
    #     sampled_seqs = outputs.get("seqs", [])

    #     # AF3 input/output dirs
    #     af3_input_dir = Path(log_dir_i, "af3_inputs")
    #     af3_pred_dir = Path(log_dir_i, "af3_preds")
    #     af3_input_dir.mkdir(parents=True, exist_ok=True)
    #     af3_pred_dir.mkdir(parents=True, exist_ok=True)

    #     # Self-consistency metrics per sample
    #     id_to_metrics = {}

    #     # Pair samples and sequences if available
    #     if sampled_seqs and len(sampled_seqs) == len(sampled_cifs):
    #         pair_iter = zip(sampled_cifs, sampled_seqs)
    #     else:
    #         # Derive sequence from CIF if not present (fallback: parse sequence from PDB features)
    #         pair_iter = []
    #         for cif_path in sampled_cifs:
    #             feats = data_utils.load_feats_from_pdb(cif_path)
    #             aatypes = feats["aatype"]
    #             # Convert to single chain string; if multiple chains, split by chain_index boundaries
    #             chain_index = feats["chain_index"] if "chain_index" in feats else torch.zeros_like(aatypes)
    #             chain_ids = chain_index.unique(sorted=True).tolist()
    #             chain_seqs = []
    #             for cid in chain_ids:
    #                 mask = (chain_index == cid)
    #                 aatype_chain = aatypes[mask]
    #                 from allatom_design.data import residue_constants as rc
    #                 seq = "".join([rc.restypes_with_x[x] for x in aatype_chain])
    #                 chain_seqs.append(seq)
    #             pair_iter.append((cif_path, ":".join(chain_seqs)))

    #     # Run AF3 per sample and compute metrics
    #     for cif_path, seq_str in tqdm(list(pair_iter), desc="AF3 predicting & scoring", leave=False):
    #         sample_id = Path(cif_path).stem
    #         job_name = sample_id

    #         chain_seqs = seq_str.split(":") if seq_str else []
    #         af3_json = _make_af3_single_json(job_name=job_name, chain_seqs=chain_seqs, model_seeds=af3_model_seeds)
    #         json_path = Path(af3_input_dir, f"{job_name}.json")
    #         with open(json_path, "w") as f:
    #             json.dump(af3_json, f)

    #         out_dir = Path(af3_pred_dir, job_name)
    #         out_dir.mkdir(parents=True, exist_ok=True)

    #         # Run AF3
    #         try:
    #             _run_af3(str(json_path), str(out_dir), runner_path=af3_runner_path, extra_args=af3_extra_args)
    #         except subprocess.CalledProcessError as e:
    #             print(f"AF3 failed for {job_name}: {e}")
    #             continue

    #         # Find predicted structure file
    #         pred_struct_path = _find_best_pred_path(str(out_dir), job_name)
    #         if pred_struct_path is None:
    #             print(f"No AF3 predicted structure found for {job_name}")
    #             continue

    #         # Compute self-consistency metrics
    #         temp_dir = str(Path(log_dir_i, "tmp"))
    #         try:
    #             metrics = _compute_sc_metrics(cif_path, pred_struct_path, temp_dir)
    #         except Exception as e:
    #             print(f"Metric computation failed for {job_name}: {e}")
    #             continue

    #         id_to_metrics[sample_id] = metrics

    #     # Save metrics CSV
    #     if id_to_metrics:
    #         metrics_df = pd.DataFrame([{"record_id": rid, **m} for rid, m in id_to_metrics.items()])
    #         metrics_df.to_csv(f"{log_dir_i}/sc_metrics_af3.csv", index=False)

    #         # Aggregate
    #         sc_metrics = defaultdict(list)
    #         for _, metrics in id_to_metrics.items():
    #             for k, v in metrics.items():
    #                 sc_metrics[k].append(v)

    #         out_metrics = {f"seq_des/mean/{k}": np.nanmean(v) for k, v in sc_metrics.items() if k != "record_id"}
    #         out_metrics.update({f"seq_des/median/{k}": np.nanmedian(v) for k, v in sc_metrics.items() if k != "record_id"})

    #         if not cfg.wandb.no_wandb:
    #             out_metrics["trainer/global_step"] = global_step
    #             out_metrics["trainer/epoch"] = epoch
    #             wandb.log(out_metrics, step=global_step)

    #     # Cleanup temp dirs to save space
    #     for d in [Path(log_dir_i, "tmp")]:
    #         if Path(d).exists():
    #             shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()



