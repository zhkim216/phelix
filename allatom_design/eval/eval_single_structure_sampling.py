import os
from pathlib import Path
import hydra
import lightning as L
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from allatom_design.eval import sampling_utils, eval_metrics
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser
from allatom_design.data.data import load_feats_from_pdb, process_single_pdb
from allatom_design.eval.folding_utils import get_struct_pred_model
import allatom_design.data.conditioning_labels as cl

@hydra.main(config_path="../configs/eval", config_name="eval_single_structure_sampling", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating the inverse folding capabilities of a denoiser model during its training run.

    We refer to "sequence recovery" as opposed to "sequence accuracy" for evaluating median across sequences rather than mean across residues.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    #make output directory
    train_dir = os.path.dirname(cfg.checkpoint_path)
    log_dir = Path(train_dir, 'eval_single_structure_sampling')  # base log dir
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config cfg
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load denoiser model
    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.checkpoint_path).eval()

    # Set up sidechain diffusion inputs
    t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)  # sidechain diffusion time

    # create sidechain diffusion noise schedule
    noise_schedule = NoiseSchedule(cfg.scn_diffusion.noise_schedule)

    # create sidechain diffusion churn config
    churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
    scd_inputs = {"num_steps": cfg.scn_diffusion.num_steps,
                  "timesteps": None,  # filled in based on batch size
                  "noise_schedule": noise_schedule,
                  "churn_cfg": churn_cfg,
                  "return_scn_diffusion_aux": False
                  }


    #load input PDB
    data = load_feats_from_pdb(cfg.pdb_path)
    batch = process_single_pdb(data)
    x, aatype, seq_mask, missing_atom_mask, residue_index, chain_index = batch["x"].to(device), batch['aatype'].to(device), batch["seq_mask"].to(device), batch["missing_atom_mask"].to(device), batch["residue_index"].to(device), batch["chain_index"].to(device)

    #repeat batch objects along new batch dimension
    B = cfg.batch_size
    x_batched, aatype_batched, seq_mask_batched, missing_atom_mask_batched, residue_index_batched, chain_index_batched = x[None,...].repeat(B,1,1,1), aatype[None,...].repeat(B,1), seq_mask[None,...].repeat(B,1), missing_atom_mask[None,...].repeat(B,1), residue_index[None, ...].repeat(B,1), chain_index[None,...].repeat(B,1)

    #handle partial sequence and sidechain conditioning
    aatype_override_mask, scn_override_mask = None, None
    if cfg.fixed_pos is not None:
        aatype_override_mask, scn_override_mask = torch.zeros_like(seq_mask_batched), torch.zeros_like(seq_mask_batched)
        aatype_override_mask[:,cfg.fixed_pos], scn_override_mask[:,cfg.fixed_pos] = 1, 1

    print(f"Evaluating with num denoising steps S={cfg.num_steps}")
    cfg.timestep_schedule.num_steps = cfg.num_steps
    t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)

    num_batches = cfg.num_samples // B + 1
    for i, batch in tqdm(enumerate(range(num_batches))):
        x, aatype, seq_mask, missing_atom_mask, residue_index, chain_index = x_batched.clone(), aatype_batched.clone(), seq_mask_batched.clone(), missing_atom_mask_batched.clone(), residue_index_batched.clone(), chain_index_batched.clone()

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
            temperature=cfg.temperature,
            aatype_decoding_order_mode=cfg.aatype_decoding_order_mode,
            num_corrector_steps=cfg.num_corrector_steps,
            corrector_step_ratio=cfg.corrector_step_ratio,
            aatype_override_mask=aatype_override_mask,
            scn_override_mask=scn_override_mask,
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

        pdbs = [f"{cfg.sample_pdb_out_dir}/batch_{i}_sample_{j}.pdb" for j in range(B)]
        SeqDenoiser.save_samples_to_pdb(samples, pdbs)

        ###predict structure to evaluate sc-RMSD and confidence
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

        codes_sc_info = eval_metrics.run_self_consistency_eval( #TODO: Add confidence, plddt, and seq_id
            pdbs,
            None, None,  # no MPNN model for co-design eval
            struct_pred_model,
            device,
            out_dir=cfg.predicted_pdb_out_dir,
            eval_codesign=True,
            temp_dir=f"{cfg.predicted_pdb_out_dir}/tmp"
        )

        # Aggregate results
        codes_metrics = defaultdict(list)
        for pdb in pdbs:
            for k, v in codes_sc_info[pdb]["sc_metrics"].items():
                codes_metrics[f"codes_{k}"].append(v.item())

        out_df = pd.DataFrame(codes_metrics)
        out_df.to_csv('out.csv')

if __name__ == "__main__":
    main()
