"""Hydra entry point for Glide docking evaluation.

Usage:
    python -m allatom_design.eval.glide.run_glide_eval
    python -m allatom_design.eval.glide.run_glide_eval schrodinger.schrodinger_path=/path/to/schrodinger
"""

from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.eval.eval_utils.eval_setup_utils import get_pdb_files, wandb_setup
from allatom_design.eval.glide.pipeline import run_glide_evaluation


def build_reference_map(
    reference_dir: str | None,
    sample_paths: list[str],
) -> dict[str, str] | None:
    """Build a mapping from sample_id to reference CIF path.

    Looks for reference files matching the sample's CCD code prefix
    in the reference directory.
    """
    if not reference_dir or not Path(reference_dir).is_dir():
        return None

    ref_files = {p.stem: str(p) for p in Path(reference_dir).glob("*.cif")}
    if not ref_files:
        return None

    ref_map = {}
    for sample_path in sample_paths:
        sample_id = Path(sample_path).stem
        # Try exact match first
        if sample_id in ref_files:
            ref_map[sample_id] = ref_files[sample_id]
            continue

        # Try matching by CCD code prefix (e.g., "0H7_len_150_0" matches "0H7_*")
        parts = sample_id.split("_")
        if parts:
            ccd_code = parts[0]
            for ref_id, ref_path in ref_files.items():
                if ref_id.startswith(ccd_code):
                    ref_map[sample_id] = ref_path
                    break

    return ref_map if ref_map else None


@hydra.main(
    config_path="../../configs_local/eval/glide",
    config_name="run_glide_eval",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    """Run Glide docking evaluation on AF3 predicted structures."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Debug mode adjustments
    if cfg.debug:
        cfg.wandb.project = f"debug_{cfg.wandb.project}"
        cfg.exp_name = f"debug_{cfg.exp_name}"

    # Setup logging directory
    log_dir = Path(
        wandb_setup(
            base_out_dir=cfg.base_out_dir,
            exp_name=cfg.exp_name,
            cfg_dict=cfg_dict,
            **cfg.wandb,
        )
    )

    # Save config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # ================================================================
    # Phase 1: Load sample paths
    # ================================================================
    print("\n" + "=" * 80)
    print("Phase 1: Loading sample paths")
    print("=" * 80 + "\n")

    sample_paths = get_pdb_files(**cfg.pdb_cfg)

    if cfg.debug:
        sample_paths = sample_paths[: cfg.num_debug_samples]

    print(f"Loaded {len(sample_paths)} samples")

    # ================================================================
    # Phase 2: Build reference map (optional)
    # ================================================================
    reference_map = None
    ref_cfg = cfg.get("reference", {})
    if ref_cfg:
        denovo_dir = ref_cfg.get("denovo_dir")
        native_dir = ref_cfg.get("native_dir")
        ref_dir = denovo_dir or native_dir
        if ref_dir:
            reference_map = build_reference_map(ref_dir, sample_paths)
            if reference_map:
                print(f"Built reference map: {len(reference_map)} matches")

    # ================================================================
    # Phase 3: Run Glide evaluation
    # ================================================================
    print("\n" + "=" * 80)
    print("Phase 3: Running Glide evaluation")
    print("=" * 80 + "\n")

    results_df = run_glide_evaluation(
        sample_paths=sample_paths,
        cfg=cfg,
        log_dir=str(log_dir),
        reference_map=reference_map,
    )

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"Results: {log_dir}/glide_results.csv")
    print(f"Total samples: {len(results_df)}")

    error_mask = results_df.get("error", pd.Series(dtype=str)).notna()
    if hasattr(error_mask, "sum"):
        n_failed = error_mask.sum()
        if n_failed > 0:
            print(f"Failed samples: {n_failed}")

    # Print score summary
    for col in ["inplace_best_docking_score", "redock_best_docking_score"]:
        if col in results_df.columns:
            valid = results_df[col].dropna()
            if len(valid) > 0:
                print(f"{col}: mean={valid.mean():.3f}, median={valid.median():.3f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    import pandas as pd  # noqa: F811
    main()
