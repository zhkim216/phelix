"""
Utils for sampling from FAMPNN.
"""
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import load_feats_from_pdb, pad_to_max_len
from allatom_design.data.datasets.sd_dataset import process_single_pdb
from allatom_design.eval import sampling_utils
from allatom_design.eval.proteinmpnn_utils import load_mpnn
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.seq_denoiser.denoisers.fampnn_denoiser import \
    FAMPNNDenoiser
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


def get_fampnn_batch(pdb_batch_files: List[str], device: str) -> Tuple[Dict[str, TensorType["b n ..."]],
                                                                       List[str],
                                                                       List[Dict[str, int]],  # maps chain letters to chain index
                                                                       ]:
    # Load and process all PDBs in this batch
    batch_list = []
    batch_chain_id_mapping = []
    for pdb_file in pdb_batch_files:
        data = load_feats_from_pdb(pdb_file)
        single = process_single_pdb(data)
        batch_list.append(single)

        # store chain ID mapping for parsing fixed positions
        batch_chain_id_mapping.append(data["chain_id_mapping"])

    pdb_names = [Path(pdb_file).stem for pdb_file in pdb_batch_files]

    # Create a batch dictionary from batch_list by stacking
    model_input_keys = ["x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index", "interface_residue_mask"]
    max_len = max(b["x"].shape[0] for b in batch_list)  # determine the max_len (max number of residues across the batch)
    batch_list = [pad_to_max_len({k: b[k].unsqueeze(0) for k in model_input_keys}, max_len)for b in batch_list]  # pad each batch to max length
    batch = {k: torch.cat([b[k] for b in batch_list], dim=0) for k in model_input_keys}  # stack the padded batches

    # Move to device
    batch = {k: batch[k].to(device) for k in model_input_keys}

    return batch, pdb_names, batch_chain_id_mapping



def create_fampnn_embeddings(model: FAMPNNDenoiser,
                             pdb_paths: List[str],
                             backbone_only: bool,
                             batch_size: int,
                             device: str,
                             out_dir: str):
    """
    Create FAMPNN embeddings for a list of PDB files.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=len(pdb_paths), desc="Creating FAMPNN embeddings")
    for i in range(0, len(pdb_paths), batch_size):
        pdb_batch_files = pdb_paths[i:i + batch_size]
        B = len(pdb_batch_files)

        batch, pdb_names, _ = get_fampnn_batch(pdb_batch_files, device)
        with torch.no_grad():
            x, aatype, seq_mask, missing_atom_mask, residue_index, chain_index = batch["x"], batch["aatype"], batch["seq_mask"], batch["missing_atom_mask"], batch["residue_index"], batch["chain_index"]
            if backbone_only:
                # Zero out aatype and sidechains
                aatype = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"]) * seq_mask.long()
                seq_mlm_mask = torch.zeros_like(seq_mask)
                scn_mlm_mask = torch.zeros_like(seq_mask)
            else:
                seq_mlm_mask = torch.ones_like(seq_mask)
                scn_mlm_mask = torch.ones_like(seq_mask)

            _, mpnn_feature_dict = model.score(x=x,
                                               aatype=aatype,
                                               seq_mask=seq_mask,
                                               missing_atom_mask=missing_atom_mask,
                                               scn_mlm_mask=scn_mlm_mask,
                                               residue_index=residue_index,
                                               chain_index=chain_index,
                                               return_embeddings=True)

        # Save FAMPNN feature dict to output directory
        mpnn_feature_dict = {k: v.cpu() for k, v in mpnn_feature_dict.items() if k in ["h_V", "h_V_enc"]}  # prune to node embeddings only
        lengths = seq_mask.sum(dim=-1).long()
        for j in range(B):
            name = pdb_names[j]
            out_file = Path(out_dir) / f"{name}.pt"
            length_j = lengths[j].item()
            mpnn_feature_dict_j = {k: v[j, :length_j].clone() for k, v in mpnn_feature_dict.items()}
            torch.save(mpnn_feature_dict_j, out_file)

        pbar.update(B)
    pbar.close()



def run_fampnn(model: SeqDenoiser, pdb_paths: List[str], device: str, cfg: DictConfig
               ) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Given a list of PDB files, run FAMPNN on them.

    Returns a dictionary mapping from PDB paths to dictionaries containing samples for that PDB, including keys:
    - x_denoised: denoised coordinates
    - seq_mask: sequence mask
    - missing_atom_mask: missing atom mask
    - residue_index: residue index
    - chain_index: chain index
    - pred_aatype: predicted amino acid types
    - pred_seq: predicted sequences

    """
    # Setup sidechain diffusion inputs
    t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)
    noise_schedule = NoiseSchedule(cfg.scn_diffusion.noise_schedule)
    churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
    scd_inputs_template = {
        "num_steps": cfg.scn_diffusion.num_steps,
        "timesteps": None,  # will be filled per batch
        "noise_schedule": noise_schedule,
        "churn_cfg": churn_cfg,
        "return_scn_diffusion_aux": False
    }

    cfg.timestep_schedule.num_steps = cfg.num_steps
    t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)

    # Process PDBs in batches of size B
    pdb_paths_repeated = np.repeat(pdb_paths, cfg.num_seqs_per_pdb)
    pbar = tqdm(total=len(pdb_paths_repeated), desc=f"FAMPNN: sampling S={cfg.num_steps} steps, {len(pdb_paths)} PDBs, {cfg.num_seqs_per_pdb} sequences per PDB...")

    pdb_to_samples = defaultdict(list)  # maps from a given pdb to all samples
    for i in range(0, len(pdb_paths_repeated), cfg.batch_size):
        pdb_batch_files = pdb_paths_repeated[i:i+cfg.batch_size]
        B = len(pdb_batch_files)
        batch, _, _ = get_fampnn_batch(pdb_batch_files, device=device)

        # Prepare scd_inputs for this batch
        scd_inputs = dict(scd_inputs_template)
        scd_inputs["timesteps"] = t_scd[None].expand(B, -1).to(device)

        # Prepare sampling timesteps
        timesteps = t_seq[None].expand(B, -1).to(device)

        cond_labels_in = None  # this is unused
        aatype_override_mask, scn_override_mask = torch.zeros_like(batch["residue_index"]), torch.zeros_like(batch["residue_index"])  # fixing positions is not implemented here

        # Run sampling
        x_denoised, aatype_denoised, aux = model.sample(
            batch["x"],
            aatype=batch["aatype"],
            seq_mask=batch["seq_mask"],
            missing_atom_mask=batch["missing_atom_mask"],
            residue_index=batch["residue_index"],
            chain_index=batch["chain_index"],
            cond_labels=cond_labels_in,
            timesteps=timesteps,
            aatype_decoding_order_mode=cfg.aatype_decoding_order_mode,
            seq_only=cfg.seq_only,
            temperature=cfg.temperature,
            repack_last=cfg.repack_last,
            psce_threshold=cfg.psce_threshold,
            noise_labels=cfg.noise_labels,
            aatype_override_mask=aatype_override_mask,
            scn_override_mask=scn_override_mask,
            omit_aas=cfg.omit_aas,
            scd_inputs=scd_inputs,
        )

        samples = {
            "x_denoised": x_denoised,
            "seq_mask": batch["seq_mask"],
            "missing_atom_mask": batch["missing_atom_mask"],
            "residue_index": batch["residue_index"],
            "chain_index": batch["chain_index"],
            "pred_aatype": aatype_denoised,
            "psce": aux["psce"],
            "seq_probs": aux["seq_probs"],
            # save other useful info
            "original_aatype": batch["aatype"],
            "interface_residue_mask": batch["interface_residue_mask"],
            "aatype_override_mask": aatype_override_mask,
            "scn_override_mask": scn_override_mask,

        }
        samples = {k: v.cpu() for k, v in samples.items()}

        # Store samples and remove padding
        for j in range(B):
            length_j = batch["seq_mask"][j].sum().long().item()
            sample_j = {k: v[j, :length_j].clone() for k, v in samples.items()}
            pdb_path = pdb_batch_files[j]
            pdb_to_samples[pdb_path].append(sample_j)

        pbar.update(B)
    pbar.close()

    # For each pdb, aggregate all FAMPNN samples
    preds = defaultdict(dict)
    for pdb, samples_list in pdb_to_samples.items():
        for k in samples_list[0].keys():
            preds[pdb][k] = torch.stack([s[k] for s in samples_list])

        # Get sampled sequences for this PDB as a list of strings
        aatype_denoised = preds[pdb]["pred_aatype"]

        pred_seqs = []
        for i in range(aatype_denoised.shape[0]):
            pred_seq = "".join([rc.restypes_with_x[aatype_denoised[i, j]] for j in range(aatype_denoised.shape[1])])
            pred_seqs.append(pred_seq)
        preds[pdb]["pred_seqs"] = pred_seqs


    return preds


def get_seq_des_model(cfg: DictConfig, device: str) -> Dict[str, Any]:
    """
    Load in a sequence design model. Similar to get_struct_pred_model()
    Example config:

    seq_des_cfg:
    # MPNN args
    model_name: "fampnn"  # ["proteinmpnn", "fampnn"]
    proteinmpnn:
        mpnn_cfg: allatom_design/configs/seq_des/proteinmpnn.yaml
        mpnn_params_dir: /media/scratch/huang_lab/allatom_design/model_params/mpnn
        overrides:
        # num seqs per structure will be batch_size * number_of_batches
        batch_size: 1
        number_of_batches: 1
        verbose: false
    fampnn:
        # FAMPNN args
        fampnn_cfg: allatom_design/configs/seq_des/fampnn.yaml
        fampnn_ckpt:
    """
    model_name = cfg.model_name
    seq_des_model = {"model_name": model_name, "cfg": cfg, "device": device}

    if model_name == "fampnn":
        fampnn_cfg = OmegaConf.load(cfg.fampnn.fampnn_cfg)
        fampnn_cfg = OmegaConf.merge(fampnn_cfg, cfg.fampnn.overrides)
        lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.fampnn.fampnn_ckpt).eval()
        seq_des_model["fampnn_model"] = lit_sd_model.model
        seq_des_model["fampnn_cfg"] = fampnn_cfg

    elif model_name == "proteinmpnn":
        mpnn_cfg = OmegaConf.load(cfg.proteinmpnn.mpnn_cfg)
        mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.proteinmpnn.overrides)  # override base mpnn config with mpnn.overrides
        mpnn_model = load_mpnn(cfg.proteinmpnn.mpnn_params_dir, mpnn_cfg, device=device)
        seq_des_model["mpnn_model"] = mpnn_model
        seq_des_model["mpnn_cfg"] = mpnn_cfg

    return seq_des_model
