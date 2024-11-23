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
from allatom_design.eval import scoring_utils, sampling_utils, multichain_scoring_utils
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

    #make output directory
    train_dir = os.path.dirname(cfg.checkpoint_path)
    log_dir = Path(train_dir, 'eval_interface_redesign')  # base log dir
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config cfg
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = FitDataset(cfg.data)
    val_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=False, drop_last=False)
    
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
                  "autoguidance_cfg": dict(cfg.scn_diffusion.autoguidance_cfg),
                  "return_scn_diffusion_aux": False
                  }
            
    for batch in tqdm(val_dataloader, desc="Evaluating fitness", leave=False):

if __name__ == "__main__":
    main()
