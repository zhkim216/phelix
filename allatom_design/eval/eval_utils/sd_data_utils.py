from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from atomworks.io.parser import parse as aw_parse
from biotite.structure import AtomArray
from omegaconf import DictConfig, OmegaConf

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None  # type: ignore[assignment]

    def delayed(func):
        raise ImportError("joblib is required when using parallel SD featurization") from None


from allatom_design.data.data import to
from allatom_design.data.datasets.atomworks_sd_dataset import sd_collator
from allatom_design.data.transform.preprocess import preprocess_transform, preprocess_transform_designed_samples
from allatom_design.data.transform.sd_featurizer import (
    featurizer_af3_prediction,
    featurizer_designed_samples,
    sd_featurizer_for_design,
)
from allatom_design.data.transform.sd_featurizer_pocket_only import sd_featurizer_pocket_only_for_design
from allatom_design.eval.eval_utils.eval_setup_utils import get_pdb_files
# from allatom_design.utils.atom_array_utils import add_pn_unit_iid_annotation
from allatom_design.utils.sample_io_utils import load_example_with_parse


def create_sample_dict(
    *,
    sample_paths: list[str] | None = None,
    sample_ids: list[str] | None = None,
    prefix: str = "input",
) -> dict[str, dict[str, str]]:
    """
    Build a sample dictionary keyed by sample ID.

    Each entry contains:
    - `{prefix}_sample_path`: source structure path
    - `{prefix}_sample_id`: sample identifier
    """
    if sample_paths is None:
        sample_paths = []

    if sample_ids is None:
        sample_ids = [Path(sample_path).stem for sample_path in sample_paths]

    sample_dict: dict[str, dict[str, str]] = defaultdict(dict)
    for i, sample_id in enumerate(sample_ids):
        sample_dict[sample_id][f"{prefix}_sample_path"] = sample_paths[i]
        sample_dict[sample_id][f"{prefix}_sample_id"] = sample_ids[i]
    return sample_dict


def prepare_sample_dict(
    cfg: DictConfig | None = None,
    sampling_inputs_df: pd.DataFrame | None = None,
    prefix: str = "input",
) -> dict[str, dict[str, str]]:
    """
    Resolve input structure files from `cfg.pdb_cfg` and package them into `sample_dict`.

    Notes:
    - Uses `get_pdb_files(**cfg.pdb_cfg)` as the canonical source of sample paths.
    - In debug mode, keeps a tiny subset and applies the existing hardcoded debug CIF behavior.
    - `sampling_inputs_df` is accepted for API compatibility with current call sites.
    """
    del sampling_inputs_df

    if cfg is None:
        raise ValueError("cfg must be provided")
    
    sample_paths = get_pdb_files(**cfg.pdb_cfg)

    if cfg.debug:
        sample_paths = sample_paths[:cfg.num_debug_samples]        

    return create_sample_dict(sample_paths=sample_paths, prefix=prefix)


def get_sd_batch(
    pdb_paths: list[str] | None = None,
    *,    
    sample_is_designed: bool = False,
    cif_parse_cfg: DictConfig | dict[str, Any] | None = None,
    preprocess_cfg: DictConfig | dict[str, Any] | None = None,
    featurizer_cfg: DictConfig | dict[str, Any] | None = None,
    device: str | None = None,
    parallel_pool: Parallel | None = None,
    sampling_inputs_df: pd.DataFrame | None = None,    
    pocket_only: bool = False,
    pocket_featurizer_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Given a list of pdb file paths, return a batch of sequence design model features.
    """
    if pdb_paths is None:
        return {}
    
    if parallel_pool is None:
        batch_examples = [
            get_sd_example(
                pdb_path=pdb_path,                
                cif_parse_cfg=cif_parse_cfg,
                preprocess_cfg=preprocess_cfg,
                featurizer_cfg=featurizer_cfg,
                sampling_inputs_df=sampling_inputs_df,
                sample_is_designed=sample_is_designed,
                pocket_only=pocket_only,
                pocket_featurizer_cfg=pocket_featurizer_cfg,
            )
            for pdb_path in pdb_paths
        ]
    else:
        batch_examples = parallel_pool(
            delayed(get_sd_example)(
                pdb_path=pdb_path,                
                cif_parse_cfg=cif_parse_cfg,
                preprocess_cfg=preprocess_cfg,
                featurizer_cfg=featurizer_cfg,
                sampling_inputs_df=sampling_inputs_df,
                sample_is_designed=sample_is_designed,
                pocket_only=pocket_only,
                pocket_featurizer_cfg=pocket_featurizer_cfg,
            )
            for pdb_path in pdb_paths
        )

    batch = sd_collator(batch_examples)
    batch = to(batch, device)
    return batch


def get_sd_example(
    pdb_path: str | None = None,    
    *,
    sample_is_designed: bool = False,
    cif_parse_cfg: DictConfig | dict[str, Any] | None = None,
    preprocess_cfg: DictConfig | dict[str, Any] | None = None,
    featurizer_cfg: DictConfig | dict[str, Any] | None = None,
    sampling_inputs_df: pd.DataFrame | None = None,    
    pocket_only: bool = False,
    pocket_featurizer_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Given a pdb file path, return a dictionary of sequence design model features.
    """
    if pdb_path is None:
        raise ValueError("pdb_path must be provided")
        
    example = load_example_with_parse(pdb_path, cif_parse_cfg)    
    
    example = preprocess_input(
        example=example,
        preprocess_cfg=preprocess_cfg,
        sample_is_designed=sample_is_designed,
    )
        
    pdb_id = Path(pdb_path).stem.split("_")[0]
    example["query_pn_unit_iids"] = resolve_query_pn_unit_iids(
        atom_array=example["atom_array"],
        sampling_inputs_df=sampling_inputs_df,
        pdb_id=pdb_id,
    )

    if pocket_only:
        pocket_cfg = OmegaConf.to_container(pocket_featurizer_cfg, resolve=True)
        featurizer = sd_featurizer_pocket_only_for_design(**pocket_cfg)
    else:
        featurizer_cfg = OmegaConf.to_container(featurizer_cfg, resolve=True)
        featurizer = sd_featurizer_for_design(**featurizer_cfg, sample_is_designed=sample_is_designed)

    return featurizer(example)


def prepare_af3_prediction(
    pdb_path: str | None = None,        
    cif_parse_cfg: DictConfig | dict[str, Any] | None = None,
    preprocess_cfg: DictConfig | dict[str, Any] | None = None,
    featurizer_cfg: DictConfig | dict[str, Any] | None = None,    
) -> dict[str, Any]:
    """
    Given a pdb file path from AF3 prediction, return sequence design model features.
    """        
                
    example = load_example_with_parse(pdb_path, cif_parse_cfg)

    example = preprocess_input(        
        example=example,        
        preprocess_cfg=preprocess_cfg,
        sample_is_designed=True,
    )
    
    featurizer_cfg = OmegaConf.to_container(featurizer_cfg, resolve=True)
    featurizer = featurizer_af3_prediction(**featurizer_cfg)
    return featurizer(example)


def prepare_designed_sample(
    pdb_path: str | None = None,        
    cif_parse_cfg: DictConfig | dict[str, Any] | None = None,
    preprocess_cfg: DictConfig | dict[str, Any] | None = None,
    featurizer_cfg: DictConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Given a pdb path from designed samples, return sequence design model features.
    """
    
    example = load_example_with_parse(pdb_path, cif_parse_cfg)

    example = preprocess_input(
        example=example,
        preprocess_cfg=preprocess_cfg,
        sample_is_designed=True,
    )

    featurizer_cfg = OmegaConf.to_container(featurizer_cfg, resolve=True)
    featurizer = featurizer_designed_samples(**featurizer_cfg)
    return featurizer(example)

def preprocess_input(
    example: dict[str, Any],
    preprocess_cfg: DictConfig | dict[str, Any] | None = None,
    sample_is_designed: bool = False,
) -> dict[str, Any]:
    """
    Preprocess an already-loaded example using SD preprocess transforms.
    """
    preprocess_cfg = OmegaConf.to_container(preprocess_cfg, resolve=True)

    if sample_is_designed:
        pipeline = preprocess_transform_designed_samples(**preprocess_cfg)
    else:
        pipeline = preprocess_transform(**preprocess_cfg)

    return pipeline(example)


def parse_query_pn_unit_iids(raw_value: Any) -> list[str]:
    """
    Parse query_pn_unit_iids from a CSV/metadata cell into a normalized list[str].
    """
    if raw_value is None:
        return []

    if isinstance(raw_value, (float, np.floating)) and np.isnan(raw_value):
        return []

    parsed = raw_value
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped == "":
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = stripped

    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()

    if isinstance(parsed, (list, tuple, set)):
        return [str(x) for x in parsed if str(x) != ""]

    return [str(parsed)] if str(parsed) != "" else []


def resolve_query_pn_unit_iids(
    *,
    atom_array: AtomArray,
    sampling_inputs_df: pd.DataFrame | None = None,
    pdb_id: str | None = None,
) -> list[str]:
    """
    Resolve query pn_unit_iids from sampling_inputs_df if available; otherwise fallback to all unique pn_unit_iid.
    """
    if (
        sampling_inputs_df is not None
        and pdb_id is not None
        and "pdb_id" in sampling_inputs_df.columns
        and "query_pn_unit_iids" in sampling_inputs_df.columns
    ):
        pdb_id_normalized = str(pdb_id).lower()
        matched = sampling_inputs_df[sampling_inputs_df["pdb_id"].astype(str).str.lower() == pdb_id_normalized]
        if not matched.empty:
            parsed = parse_query_pn_unit_iids(matched["query_pn_unit_iids"].iloc[0])
            if len(parsed) > 0:
                return parsed

    if "pn_unit_iid" in atom_array.get_annotation_categories():
        return [str(x) for x in np.unique(atom_array.pn_unit_iid).tolist()]

    raise ValueError("pn_unit_iid annotation is required")


def resolve_selectivity_row(
    *,
    sampling_inputs_df: pd.DataFrame,
    pdb_id: str,
    guidance_direction: int,
) -> dict[str, Any]:
    """Resolve one backbone's selectivity-assay context from the paired CSV.

    The paired CSV schema has columns `pdb_id_{1,2}`, `query_pn_unit_iids_{1,2}`,
    `ccd_code_{1,2}`. A single `pdb_id` may appear at either `_1` (the H-bond-rich
    position) or `_2` (the H-bond-poor position) across rows; this function
    locates it and returns the self/partner pair plus the guidance target.

    Args:
        sampling_inputs_df: DataFrame loaded from the paired selectivity CSV.
        pdb_id: Backbone identifier (case-insensitive).
        guidance_direction: 1 or 2. Selects `ccd_code_{guidance_direction}` as
            the guidance target — independent of which slot the backbone
            occupies. One pass with `guidance_direction=1` designs every
            backbone with the potential pulling toward the H-bond-rich CCD;
            `guidance_direction=2` pulls toward the H-bond-poor CCD.

    Returns:
        dict with keys:
            pdb_id_self, query_pn_unit_iids_self, ccd_self,
            pdb_id_partner, query_pn_unit_iids_partner, ccd_partner,
            guidance_target_ccd, pocket_subcluster_id,
            self_position (1 or 2).

    Raises:
        ValueError: if `guidance_direction` is not in {1, 2}, required columns
            are missing, or `pdb_id` is absent from both slots.
    """
    if guidance_direction not in (1, 2):
        raise ValueError(f"guidance_direction must be 1 or 2, got {guidance_direction}")

    required_cols = {
        "pdb_id_1", "pdb_id_2",
        "query_pn_unit_iids_1", "query_pn_unit_iids_2",
        "ccd_code_1", "ccd_code_2",
    }
    missing = required_cols - set(sampling_inputs_df.columns)
    if missing:
        raise ValueError(f"sampling_inputs_df missing columns: {sorted(missing)}")

    pdb_lc = str(pdb_id).lower()
    for self_pos in (1, 2):
        other_pos = 3 - self_pos
        hit = sampling_inputs_df[
            sampling_inputs_df[f"pdb_id_{self_pos}"].astype(str).str.lower() == pdb_lc
        ]
        if not hit.empty:
            row = hit.iloc[0]
            out = {
                "pdb_id_self": str(row[f"pdb_id_{self_pos}"]),
                "query_pn_unit_iids_self":
                    parse_query_pn_unit_iids(row[f"query_pn_unit_iids_{self_pos}"]),
                "ccd_self": str(row[f"ccd_code_{self_pos}"]),
                "pdb_id_partner": str(row[f"pdb_id_{other_pos}"]),
                "query_pn_unit_iids_partner":
                    parse_query_pn_unit_iids(row[f"query_pn_unit_iids_{other_pos}"]),
                "ccd_partner": str(row[f"ccd_code_{other_pos}"]),
                "guidance_target_ccd": str(row[f"ccd_code_{guidance_direction}"]),
                "self_position": self_pos,
            }
            if "pocket_subcluster_id" in sampling_inputs_df.columns:
                out["pocket_subcluster_id"] = int(row["pocket_subcluster_id"])
            return out

    raise ValueError(
        f"pdb_id={pdb_id} not found in either pdb_id_1 or pdb_id_2 column of "
        f"sampling_inputs_df (rows={len(sampling_inputs_df)})"
    )
