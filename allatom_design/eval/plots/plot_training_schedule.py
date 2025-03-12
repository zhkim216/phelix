# plot_training_schedule.py
import copy
import math
from pathlib import Path
from typing import Any, Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from scipy.stats import beta as beta_dist

from allatom_design.interpolants.ad_interpolants.sd3_rf_interpolant import \
    SD3_RF


@hydra.main(version_base=None, config_path="../../configs/eval/plots", config_name="plot_training_schedule")
def main(cfg: DictConfig) -> None:
    """
    This script will plot:
      - A histogram of sampled timesteps
      - An approximate PDF of the distribution
    for each combination of parameters in cfg.training_t_schedule_cfg_array.<schedule>.<param>.

    All histograms and PDFs will be shown on the same plot with partial transparency.
    """
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    schedule = cfg.training_t_schedule
    # We'll collect (label, t_samples, (x_domain, pdf_values)) for each param combo
    results = []

    if schedule == "logit_normal":
        loc_list = cfg.training_t_schedule_cfg_array.logit_normal.location
        scale_list = cfg.training_t_schedule_cfg_array.logit_normal.scale
        for loc, scale in zip(loc_list, scale_list):
            # Create a copy of the config so we can modify just for this run
            cfg_mod = copy.deepcopy(cfg)
            # We must recreate the training_t_schedule_cfg that SD3_RF expects
            cfg_mod.training_t_schedule_cfg = {
                "logit_normal": {
                    "location": loc,
                    "scale": scale
                },
                "mode": {},       # placeholders
                "proteina": {}
            }
            # Instantiate interpolant
            interpolant = SD3_RF(cfg_mod)

            # Sample timesteps
            n_samples = 10000
            t = interpolant.sample_timestep(n_samples, device=torch.device("cpu")).cpu().numpy()

            # The PDF for logit_normal(t) with location=loc, scale=scale:
            #   u = (logit(t) - loc) / scale -> normal(0,1) in u
            #   t in (0,1)
            #   PDF(t) = Normal_pdf( (logit(t)-loc)/scale ) / ( t*(1-t)*scale )
            #   where logit(t) = ln(t/(1-t))
            #   Normal_pdf(z) = 1/(sqrt(2*pi)) * exp(-z^2/2)
            x_vals = np.linspace(0.001, 0.999, 300)
            pdf_vals = []
            for xv in x_vals:
                # logit
                logit_xv = math.log(xv / (1. - xv))
                z = (logit_xv - loc) / scale
                normal_pdf_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
                # derivative of logit is 1/(xv*(1-xv))
                pdf = (normal_pdf_z / scale) * (1.0 / (xv * (1.0 - xv)))
                pdf_vals.append(pdf)
            pdf_vals = np.array(pdf_vals)

            label = f"logit_normal: loc={loc}, scale={scale}"
            results.append((label, t, (x_vals, pdf_vals)))

    elif schedule == "mode":
        s_list = cfg.training_t_schedule_cfg_array.mode.s
        for s_val in s_list:
            cfg_mod = copy.deepcopy(cfg)
            cfg_mod.training_t_schedule_cfg = {
                "logit_normal": {},
                "mode": {
                    "s": s_val
                },
                "proteina": {}
            }
            interpolant = SD3_RF(cfg_mod)

            n_samples = 10000
            t = interpolant.sample_timestep(n_samples, device=torch.device("cpu")).cpu().numpy()

            # The PDF for 'mode' distribution is not as standard. Based on the code:
            #   u = rand(0,1)
            #   t = 1 - u - s*(cos(pi/2 * u)^2 - 1 + u)
            # It's a direct transform, so for PDF we'd do a derivative. If you want
            # an exact PDF, you'd have to do dt/du carefully. We'll do a quick approach
            # by generating a dense set of u in [0,1], mapping to t, then histogram that
            # as an approximate PDF.

            # Approximate PDF via transform of uniform:
            u_vals = np.linspace(0.0, 1.0, 100000)
            t_vals = 1.0 - u_vals - s_val * ((np.cos(np.pi/2 * u_vals) ** 2) - 1.0 + u_vals)
            # We'll bin them to approximate the PDF in [0, 1].
            hist_t, bin_edges = np.histogram(t_vals, bins=300, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            pdf_vals = hist_t
            x_vals = bin_centers

            label = f"mode: s={s_val}"
            results.append((label, t, (x_vals, pdf_vals)))

    elif schedule == "proteina":
        alpha_list = cfg.training_t_schedule_cfg_array.proteina.alpha
        beta_list = cfg.training_t_schedule_cfg_array.proteina.beta
        for alpha, beta in zip(alpha_list, beta_list):
            cfg_mod = copy.deepcopy(cfg)
            cfg_mod.training_t_schedule_cfg = {
                "logit_normal": {},
                "mode": {},
                "proteina": {
                    "alpha": alpha,
                    "beta": beta
                }
            }
            interpolant = SD3_RF(cfg_mod)

            n_samples = 10000
            t = interpolant.sample_timestep(n_samples, device=torch.device("cpu")).cpu().numpy()

            # The mixture distribution is:
            #   With prob 0.02 -> Uniform(0,1)
            #   With prob 0.98 -> Beta(alpha, beta)
            # PDF(t) = 0.02 * 1.0 + 0.98 * Beta_pdf(t; alpha, beta)
            x_vals = np.linspace(0.0, 1.0, 300)
            pdf_vals = np.zeros_like(x_vals)

            beta_pdf_vals = beta_dist.pdf(x_vals, alpha, beta)

            pdf_vals = 0.02 + 0.98 * beta_pdf_vals

            label = f"proteina: alpha={alpha}, beta={beta}"
            results.append((label, t, (x_vals, pdf_vals)))

    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    # Plot all histograms
    plt.figure(figsize=(8, 6))
    for label, t_samples, (pdf_x, pdf_vals) in results:
        plt.hist(t_samples, bins=100, density=True, alpha=0.3, label=label)

    plt.title(f"Timestep Distribution: {schedule}")
    plt.xlabel("t")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.out_dir}/training_t_schedule_histogram.png")
    plt.close()

    # Plot all PDFs
    plt.figure(figsize=(8, 6))
    for label, _, (pdf_x, pdf_vals) in results:
        plt.plot(pdf_x, pdf_vals, linewidth=2.0, label=label)

    plt.title(f"Timestep PDF: {schedule}")
    plt.xlabel("t")
    plt.ylabel("PDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.out_dir}/training_t_schedule_pdf.png")
    plt.close()


if __name__ == "__main__":
    main()
