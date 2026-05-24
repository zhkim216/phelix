"""
Two-stage ligand-conditioned sequence design.

Supports two modes:
- pocket_first:   Stage 1 designs pocket (fix scaffold) → Stage 2 designs scaffold (fix pocket)
- scaffold_first: Stage 1 designs scaffold (fix pocket) → Stage 2 designs pocket (fix scaffold)

Each stage uses a separately specified model.

Usage:
    python -m allatom_design.eval.sampling.lc_seq_des_multi_two_stage
"""
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import yaml

from allatom_design.eval.eval_utils.eval_setup_utils import wandb_setup
from allatom_design.eval.eval_utils.sd_data_utils import prepare_sample_dict
from allatom_design.eval.eval_utils.seq_des_utils import (
    redesign_with_lcaliby_two_stage,
)
from allatom_design.eval.eval_utils.folding_utils import (
    evaluate_af3_self_consistency,
    evaluate_af3_docking_consistency,
)


@hydra.main(config_path="../../configs_local/eval/sampling", config_name="lc_seq_des_multi_two_stage", version_base="1.3.2")
def main(cfg: DictConfig):
    """Two-stage ligand-conditioned sequence design."""
    ###########################################################
    # Phase 0: Basic setup
    ###########################################################
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if cfg.debug:
        cfg.wandb.project = f"debug_{cfg.wandb.project}"
        cfg.exp_name = f"debug_{cfg.exp_name}"

    log_dir = Path(wandb_setup(base_out_dir=cfg.base_out_dir, exp_name=cfg.exp_name, cfg_dict=cfg_dict, **cfg.wandb))

    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    if cfg.sampling_inputs_csv is not None:
        sampling_inputs_df = pd.read_csv(cfg.sampling_inputs_csv)
    else:
        sampling_inputs_df = None

    ###########################################################
    # Phase 1: Prepare samples
    ###########################################################
    print("\n" + "=" * 80)
    print("Phase 1: Preparing samples")
    print("=" * 80 + "\n")

    sample_dict = prepare_sample_dict(cfg=cfg, sampling_inputs_df=sampling_inputs_df)

    array_id = cfg.pdb_cfg.get("array_id", None)
    csv_suffix = f"_array_{array_id}" if array_id is not None else ""

    ###########################################################
    # Phase 2: Two-stage design
    ###########################################################
    mode = cfg.two_stage_cfg.mode
    print("\n" + "=" * 80)
    print(f"Phase 2: Two-stage design (mode={mode})")
    print("=" * 80 + "\n")

    if not cfg.input_sample_is_designed:
        cif_parse_cfg_for_input = cfg.cif_cfg.parse.native
        preprocess_cfg_for_input = cfg.preprocess_cfg.native
    else:
        cif_parse_cfg_for_input = cfg.cif_cfg.parse.designed_samples
        preprocess_cfg_for_input = cfg.preprocess_cfg.designed_samples

    results = redesign_with_lcaliby_two_stage(
        mode=mode,
        seed=cfg.seed,
        input_sample_is_designed=cfg.input_sample_is_designed,
        sample_dict=sample_dict,
        pdb_cfg=cfg.pdb_cfg,
        pocket_seq_des_cfg=cfg.pocket_seq_des_cfg,
        scaffold_seq_des_cfg=cfg.scaffold_seq_des_cfg,
        cif_parse_cfg_for_input=cif_parse_cfg_for_input,
        cif_parse_cfg_for_designed=cfg.cif_cfg.parse.designed_samples,
        preprocess_cfg_for_input=preprocess_cfg_for_input,
        preprocess_cfg_for_designed=cfg.preprocess_cfg.designed_samples,
        featurizer_cfg=cfg.featurizer_cfg.design,
        cif_save_cfg=cfg.cif_cfg.save,
        sampling_inputs_df=sampling_inputs_df,
        log_dir=log_dir,
        pocket_distance=cfg.two_stage_cfg.pocket_distance,
        use_pseudocb=cfg.two_stage_cfg.get("use_pseudocb_for_pocket_annotation", False),
        protein_only=cfg.get("protein_only", False),
        csv_suffix=csv_suffix,
        num_workers=cfg.get("num_workers", 0),
        num_debug_samples=cfg.get("num_debug_samples", 5),
        debug=cfg.get("debug", False),
    )

    ###########################################################
    # Phase 3: AF3 evaluation (iterates over stage2 directories)
    ###########################################################
    if cfg.struct_pred_cfg.evaluate_self_consistency:
        print("\n" + "=" * 80)
        print("Phase 3a: AF3 Self-Consistency Evaluation")
        print("=" * 80 + "\n")
        for sample_dict_per_ckpt, stage2_dir, ckpt_info in results:
            print(f"\nEvaluating: {stage2_dir.name}")
            evaluate_af3_self_consistency(
                sample_dict=sample_dict_per_ckpt,
                out_dir=stage2_dir,
                struct_pred_cfg=cfg.struct_pred_cfg,
                cif_parse_cfg=cfg.cif_cfg.parse.af3_predictions,
                preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
                featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
                pocket_cfg=cfg.pocket_cfg,
                no_wandb=cfg.wandb.no_wandb,
                ckpt_info=ckpt_info,
                calculate_metrics_only=cfg.struct_pred_cfg.calculate_metrics_only,
                csv_suffix=csv_suffix,
            )

    if cfg.struct_pred_cfg.evaluate_docking_consistency:
        print("\n" + "=" * 80)
        print("Phase 3b: AF3 Docking Consistency Evaluation")
        print("=" * 80 + "\n")
        for sample_dict_per_ckpt, stage2_dir, ckpt_info in results:
            print(f"\nEvaluating (TC): {stage2_dir.name}")
            evaluate_af3_docking_consistency(
                sample_dict=sample_dict_per_ckpt,
                out_dir=stage2_dir,
                struct_pred_cfg=cfg.struct_pred_cfg,
                cif_parse_cfg=cfg.cif_cfg.parse.af3_predictions,
                preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
                featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
                pocket_cfg=cfg.pocket_cfg,
                no_wandb=cfg.wandb.no_wandb,
                ckpt_info=ckpt_info,
                calculate_metrics_only=cfg.struct_pred_cfg.calculate_metrics_only,
                csv_suffix=csv_suffix,
            )

    print("\n" + "=" * 80)
    print("All phases complete!")
    print(f"Results saved to {log_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
