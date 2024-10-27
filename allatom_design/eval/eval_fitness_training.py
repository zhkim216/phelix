import glob
import os
import re
from pathlib import Path
import hydra
import lightning as L
import torch
import wandb
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from allatom_design.data.datasets.fitness_dataset import FitDataset
from allatom_design.eval import scoring_utils, sampling_utils
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser

from scipy.stats import pearsonr, spearmanr

@hydra.main(config_path="../configs/eval", config_name="eval_fitness_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating the inverse folding capabilities of a denoiser model during its training run.

    We refer to "sequence recovery" as opposed to "sequence accuracy" for evaluating median across sequences rather than mean across residues.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)


    # Create wandb dir
    wandb_dir = str(Path(cfg.out_dir))
    Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)

    # Set wandb cache directory
    wandb_cache_dir = str(Path(cfg.out_dir, "cache", "wandb"))
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up logging
    if cfg.no_wandb:
        log_dir = Path(cfg.out_dir, "debug")
    else:
        wandb.init(
            project=cfg.project,
            entity=cfg.wandb_id,
            name=cfg.exp_name,
            group=cfg.group,
            config=cfg_dict,
            dir=wandb_dir,
        )
        log_dir = Path(cfg.out_dir, wandb.run.name)  # base log dir

    # Set up out directories
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Preserve configcfg
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get checkpoints from denoiser training run
    pattern = re.compile(r"sd-epoch\d+\.ckpt$")  # only consider ckpts of form sd-epochXXXX.ckpt
    sd_ckpts = glob.glob(f"{cfg.denoiser_train_dir}/checkpoints/*.ckpt")
    sd_ckpts = natsorted([ckpt for ckpt in sd_ckpts if pattern.search(Path(ckpt).name)])[::cfg.eval_every_n_ckpts]

    pbar = tqdm(sd_ckpts, desc="Evaluating checkpoints")

    for sd_ckpt in pbar:

        # Skip if epoch is before start_epoch
        epoch = int(Path(sd_ckpt).stem.replace("sd-epoch", ""))
        pbar.set_postfix_str(f"Epoch: {epoch}")
        if (cfg.start_epoch is not None) and (epoch < cfg.start_epoch):
                continue

        for dataset_name in cfg.datasets:
        
            cfg.data.pdb_path = f"{cfg.dataset_path}/{dataset_name}"
            cfg.data = scoring_utils.update_data_cfg(cfg.data)
            dataset = FitDataset(cfg.data)
            val_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=False, drop_last=False)
            
            # Load denoiser model
            lit_sd_model = LitSeqDenoiser.load_from_checkpoint(sd_ckpt).eval()
            
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
                          "autoguidance_cfg": dict(cfg.scn_diffusion.autoguidance_cfg),
                          "return_scn_diffusion_aux": False
                          }
            
            ### BEGIN EVAL ###
            metrics = {}
            scores_all = []
            labels_all = []

            #group examples by experiment for detailed scoring
            if cfg.data.group_by_exp:
                scores_exp = {}
                labels_exp = {}
                
            for batch in tqdm(val_dataloader, desc="Evaluating fitness", leave=False):
                pdb_key, mutations, labels, experiment, pdb_data = batch['pdb_key'], batch["mut"], batch["label"], batch["experiment"], batch["pdb_data"]
                x, aatype, seq_mask, residue_index, chain_index, confidence = pdb_data["x"].to(device), pdb_data["aatype"].to(device), pdb_data["seq_mask"].to(device), pdb_data["residue_index"].to(device), pdb_data["chain_index"].to(device), pdb_data["res_b_factors"].to(device)
                scd_inputs["timesteps"] = t_scd[None].expand(x.shape[0], -1).to(device)

                scores = scoring_utils.score_seq(lit_sd_model,
                                                 x,   
                                                 aatype,
                                                 seq_mask,
                                                 residue_index,
                                                 chain_index,
                                                 confidence,
                                                 mutations,
                                                 scd_inputs,
                                                 cfg.data.scoring_method).cpu().tolist()
   
                scores_all += scores
                labels_all += labels

                #group scores and labels by experiment for detailed scoring
                if cfg.data.group_by_exp:
                    for score, label, exp in zip(scores, labels, experiment):
                        if exp in scores_exp.keys():
                            scores_exp[exp].append(score)
                            labels_exp[exp].append(label)
                        else:
                            scores_exp[exp] = [score]
                            labels_exp[exp] = [score]

            #collect metrics
            metrics[f'{dataset_name}_pearson_r'], metrics[f'{dataset_name}_spearman_r'] = abs(pearsonr(scores_all, labels_all).correlation), abs(spearmanr(scores_all, labels_all).correlation)
            print(f"Epoch Pearson R All: {metrics[f'{dataset_name}_pearson_r'] }")
            print(f"Epoch Spearman Rho All: {metrics[f'{dataset_name}_spearman_r'] }")
            
            #group scores and labels by experiment for detailed scoring        
            if cfg.data.group_by_exp:
                metrics[f'{dataset_name}_pearson_r_avg'], metrics[f'{dataset_name}_spearman_r_avg'] = scoring_utils.get_avg_metrics(scores_exp, labels_exp)
                print(f"Epoch Pearson R Avg: {metrics[f'{dataset_name}_pearson_r_avg'] }")
                print(f"Epoch Spearman Rho Avg: {metrics[f'{dataset_name}_spearman_r_avg'] }")
            
            # Log metrics to wandb
            if not cfg.no_wandb:
                # Get global step
                global_step = torch.load(sd_ckpt, map_location="cpu")["global_step"]
                metrics["trainer/global_step"] = global_step
                metrics["trainer/epoch"] = epoch
                wandb.log(metrics, step=global_step)

if __name__ == "__main__":
    main()
