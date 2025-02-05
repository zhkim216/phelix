import math
import os
import re
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List

import hydra
import lightning as L
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.conditioning_labels as cl
from allatom_design.data.data import load_feats_from_pdb, process_single_pdb
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.folding_utils import get_struct_pred_model
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser
from allatom_design.data import residue_constants as rc


@hydra.main(config_path="../../configs/eval/sampling", config_name="sample_single", version_base="1.3.2")
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
    sample_out_dir = f"{out_dir}/samples"  # directory for model samples
    pred_out_dir = f"{out_dir}/preds"  # directory for structure predictions (if running folding)

    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    Path(pred_out_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(Path(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load denoiser model
    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.checkpoint_path).eval()

    # Load structure prediction model
    if cfg.run_self_consistency_eval:
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    # Set up sidechain diffusion inputs
    t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)  # sidechain diffusion time

    ## create sidechain diffusion noise schedule
    noise_schedule = NoiseSchedule(cfg.scn_diffusion.noise_schedule)

    ## create sidechain diffusion churn config
    churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
    scd_inputs = {"num_steps": cfg.scn_diffusion.num_steps,
                  "timesteps": None,  # filled in based on batch size
                  "noise_schedule": noise_schedule,
                  "churn_cfg": churn_cfg,
                  "return_scn_diffusion_aux": False
                  }


    # Load input PDB
    data = load_feats_from_pdb(cfg.pdb_path)
    batch = process_single_pdb(data)

    # Override sequence from PDB at certain positions and adds to fixed_pos_seq
    if cfg.pos_seq_override is not None:
        print(f"Overriding sequence at certain positions and conditioning on them: {cfg.pos_seq_override}")
        pos_seq_override = OrderedDict(cfg.pos_seq_override)  # ensure mapping is preserved
        abs_pos_override = parse_fixed_positions(",".join(pos_seq_override.keys()), data["chain_id_mapping"], batch["residue_index"], batch["chain_index"])

        for pos, aa in zip(abs_pos_override, pos_seq_override.values()):
            # change aatype at this position
            batch["aatype"][pos] = rc.restype_order[aa]

        # Add to fixed_pos_seq
        cfg.fixed_pos_seq = f"{cfg.fixed_pos_seq}," if cfg.fixed_pos_seq is not None else ""
        cfg.fixed_pos_seq += ",".join(pos_seq_override.keys())

    # Move inputs to device
    model_input_keys = ["x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index"]
    model_inputs = {k: batch[k].to(device) for k in model_input_keys}

    # Repeat inputs along new batch dimension
    B = cfg.batch_size
    repeat_along_batch_fn = lambda x: x[None, ...].repeat(B, *([1] * len(x.shape)))
    model_inputs = {k: repeat_along_batch_fn(v) for k, v in model_inputs.items()}

    # Ensure that input PDB does not have insertion codes
    if (data["insertion_code_offsets"] > 0).any():
        raise ValueError("Input PDB has insertion codes, which is not handled by fixed_pos specifications. Please renumber your input PDB before running sampling.")

    # Handle partial sequence and sidechain conditioning
    aatype_override_mask, scn_override_mask = None, None

    if cfg.fixed_pos_seq is not None:
        # sequence override
        print(f"Fixing sequence at positions {cfg.fixed_pos_seq}")
        abs_fixed_pos_seq = parse_fixed_positions(cfg.fixed_pos_seq, data["chain_id_mapping"], batch["residue_index"], batch["chain_index"])
        aatype_override_mask = torch.zeros_like(model_inputs["seq_mask"])
        aatype_override_mask[:, abs_fixed_pos_seq] = 1

        # print fixed sequence
        fixed_seq_viz = "".join([rc.restypes_with_x[batch["aatype"][i]] if aatype_override_mask[0, i] else "-" for i in range(aatype_override_mask.shape[1])])
        print(f"Fixed sequence: {fixed_seq_viz}")
    else:
        print("No fixed sequence positions specified.")

    if cfg.fixed_pos_scn is not None:
        # sidechain override
        print(f"Fixing sidechains at positions {cfg.fixed_pos_scn}")
        abs_fixed_pos_scn = parse_fixed_positions(cfg.fixed_pos_scn, data["chain_id_mapping"], batch["residue_index"], batch["chain_index"])
        scn_override_mask = torch.zeros_like(model_inputs["seq_mask"])
        scn_override_mask[:, abs_fixed_pos_scn] = 1

        if cfg.pos_seq_override is not None:
            # ensure that we're not fixing sidechains when we override the PDB sequence
            assert scn_override_mask[:, abs_pos_override].sum() == 0, "Cannot fix sidechains at positions where the sequence from the PDB is overridden."

        # print fixed sidechains
        fixed_scn_viz = "".join([rc.restypes_with_x[batch["aatype"][i]] if scn_override_mask[0, i] else "-" for i in range(scn_override_mask.shape[1])])
        print(f"Fixed sidechains: {fixed_scn_viz}")
    else:
        print("No fixed sidechain positions specified.")

    if cfg.pos_restrict_aatype is not None:
        # restrict aatype at certain positions
        print(f"Restricting aatype sampling at some positions: {cfg.pos_restrict_aatype}")
        pos_restrict_aatype = OrderedDict(cfg.pos_restrict_aatype)  # ensure mapping is preserved
        abs_restrict_pos = parse_fixed_positions(",".join(pos_restrict_aatype.keys()), data["chain_id_mapping"], batch["residue_index"], batch["chain_index"])

        B, N = model_inputs["seq_mask"].shape
        K = len(rc.restype_order_with_x)

        restrict_pos_mask = torch.zeros((B, N), dtype=torch.float32, device=device)
        allowed_aatype_mask = torch.zeros((B, N, K), dtype=torch.long, device=device)

        for abs_pos, allowed_aatypes in zip(abs_restrict_pos, pos_restrict_aatype.values()):
            # restrict aatypes at this position
            restrict_pos_mask[:, abs_pos] = 1.0

            # first, disallow all aatypes
            allowed_aatype_mask[:, abs_pos, :] = 0.0

            # only allow the specified aatypes
            for letter in allowed_aatypes:
                allowed_aatype_mask[:, abs_pos, rc.restype_order[letter]] = 1.0

        restrict_pos_aatype = (restrict_pos_mask, allowed_aatype_mask)
    else:
        restrict_pos_aatype = None


    # Sampling loop
    print(f"Evaluating with num denoising steps S={cfg.num_steps}")
    cfg.timestep_schedule.num_steps = cfg.num_steps
    t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)

    num_batches = math.ceil(cfg.num_samples // B)
    for i in tqdm(range(num_batches)):
        x, aatype, seq_mask, missing_atom_mask, residue_index, chain_index = [model_inputs[k].clone() for k in model_input_keys]

        timesteps = t_seq[None].expand(x.shape[0], -1).to(device)

        # Define sidechain diffusion timesteps
        scd_inputs["timesteps"] = t_scd[None].expand(x.shape[0], -1).to(device)

        # Define conditioning labels when we inverse fold
        cond_labels_in = {
            "crop_aug": torch.Tensor([cl.DEFAULT_TOKEN_ID['crop_aug']]*B).to(device),
            "dataset_source": torch.Tensor([cl.DEFAULT_TOKEN_ID['dataset_source']]*B).to(device),
            "designability": torch.Tensor([cl.PLACEHOLDER_TOKEN_ID]*B).to(device)
        }

        x_denoised, aatype_denoised, aux = lit_sd_model.model.sample(
            x,
            aatype=aatype,
            seq_mask=seq_mask,
            missing_atom_mask=missing_atom_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            cond_labels=cond_labels_in,
            timesteps=timesteps,
            aatype_decoding_order_mode=cfg.aatype_decoding_order_mode,
            num_corrector_steps=cfg.num_corrector_steps,
            corrector_step_ratio=cfg.corrector_step_ratio,
            seq_only=cfg.seq_only,
            temperature=cfg.temperature,
            repack_last=cfg.repack_last,
            repack_every_step=cfg.repack_every_step,
            psce_threshold=cfg.psce_threshold,
            noise_labels=cfg.noise_labels,
            aatype_override_mask=aatype_override_mask,
            scn_override_mask=scn_override_mask,
            restrict_pos_aatype=restrict_pos_aatype,
            omit_aas=cfg.omit_aas,
            scd_inputs=scd_inputs,
        )

        samples = {"x_denoised": x_denoised,
                "seq_mask": seq_mask,
                "missing_atom_mask": missing_atom_mask,
                "residue_index": residue_index,
                "chain_index": chain_index,
                "pred_aatype": aatype_denoised,
                "aatype_pred_traj": aux["aatype_pred_traj"],
                "aatype_t_traj": aux["aatype_t_traj"],
                "chain_index": chain_index, #save with same chain index as input
                "psce": aux["psce"]
        }

        pdbs = [f"{sample_out_dir}/sample_{i*B + j}.pdb" for j in range(B)]
        SeqDenoiser.save_samples_to_pdb(samples, pdbs)

        if cfg.run_self_consistency_eval:
            codes_sc_info = eval_metrics.run_self_consistency_eval( #TODO: Add confidence, plddt, and seq_id
                pdbs,
                None, None,  # no MPNN model for co-design eval
                struct_pred_model,
                device,
                out_dir=pred_out_dir,
                eval_codesign=True,
                temp_dir=f"{pred_out_dir}/tmp"
            )

            # Aggregate results
            codes_metrics = defaultdict(list)
            for pdb in pdbs:
                for k, v in codes_sc_info[pdb]["sc_metrics"].items():
                    codes_metrics[f"codes_{k}"].append(v.item())

            out_df = pd.DataFrame(codes_metrics)
            out_df.to_csv(f"{out_dir}/self_consistency_metrics_batch_{i}.csv", index=False)


def parse_fixed_positions(fixed_pos_str: str,
                          chain_id_mapping: Dict[str, int],
                          residue_index: TensorType["n", int],
                          chain_index: TensorType["n", int]) -> TensorType["k", int]:
    """
    Parse a list of fixed positions in the format ["A1", "A10-25", ...] and
    return the corresponding list of absolute indices.

    Args:
        fixed_pos_list (str): Comma-separated string representing fixed positions (e.g., "A1,A10-25").
        chain_id_mapping (dict): Mapping of chain letter to chain index (e.g., {'A': 0, 'B': 1}).
        residue_index (torch.Tensor): Tensor of residue indices (shape: [N]).
        chain_index (torch.Tensor): Tensor of chain indices (shape: [N]).

    Returns:
        List[int]: List of absolute indices to set to 1 in the masks.
    """
    fixed_indices = []

    fixed_pos_str = fixed_pos_str.strip()
    if not fixed_pos_str:
        return fixed_indices  # no positions specified

    fixed_pos_list = [item.strip() for item in fixed_pos_str.split(",") if item.strip()]

    for pos in fixed_pos_list:
        # Match pattern like "A10" or "A10-25"
        match = re.match(r"([A-Za-z])(\d+)(?:-(\d+))?$", pos)
        if not match:
            raise ValueError(f"Invalid position format: {pos}")

        chain_letter = match.group(1)
        start_residue = int(match.group(2))
        end_residue = int(match.group(3)) if match.group(3) else start_residue

        if chain_letter not in chain_id_mapping:
            raise ValueError(f"Chain ID {chain_letter} not found in mapping.")

        # For the given chain, create a mask for all residues in the desired range
        chain_i = chain_id_mapping[chain_letter]
        range_mask = (chain_index == chain_i) & (residue_index >= start_residue) & (residue_index <= end_residue)
        matching_indices = torch.where(range_mask)[0]

        # Check that each residue in the requested range; warn if not found
        found_residues = residue_index[matching_indices].tolist()
        found_residues_set = set(found_residues)

        for r in range(start_residue, end_residue + 1):
            if r not in found_residues_set:
                print(f"Warning: Requested position {chain_letter}{r} not found in structure.")

        # Extend our fixed indices with whatever we did find
        fixed_indices.extend(matching_indices.tolist())

    return list(set(fixed_indices))


if __name__ == "__main__":
    main()
