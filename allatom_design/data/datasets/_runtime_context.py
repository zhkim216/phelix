"""Runtime derivation and filtering of context_group / num_contacting_protein_chains columns.

- `derive_context_columns`: build `q_pn_unit_context_group_iids` and
  `num_contacting_protein_chains` from the raw contact list for parquets that
  don't ship them pre-computed, or when the caller wants a different distance cutoff.
- `apply_context_group_chain_type_whitelist`: restrict the derived context
  group to a user-configured subset of chain types (PMSM, PMM, branched, nucleic,
  peptide). Runs after `derive_context_columns` so it composes with any caller
  gating or pre-computed columns.
"""

from __future__ import annotations

import json
import logging

import atomworks.enums as aw_enums
import pandas as pd

logger = logging.getLogger(__name__)


def derive_context_columns(
    metadata_df: pd.DataFrame,
    context_distance: float,
    contacts_col: str = "q_pn_unit_contacting_pn_unit_iids",
    protein_col: str = "q_pn_unit_is_protein",
    iid_col: str = "q_pn_unit_iid",
) -> pd.DataFrame:
    """Add `num_contacting_protein_chains` and `q_pn_unit_context_group_iids`.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Must contain `contacts_col`, `protein_col`, and `iid_col`. `contacts_col`
        rows may be JSON strings or already-parsed list-of-dicts (each dict with
        at least `pn_unit_iid` and `min_distance` keys).
    context_distance : float
        Angstrom cutoff. Only contacts with `min_distance <= context_distance`
        are included in the context group / protein count.
    contacts_col, protein_col, iid_col : str
        Column names. Defaults match the v8 parquet schema.

    Returns
    -------
    pd.DataFrame
        New dataframe (input is not mutated) with two additional columns:
        - `q_pn_unit_context_group_iids` (list[str], always includes self iid)
        - `num_contacting_protein_chains` (int32)

    Notes
    -----
    The `pn_unit_iid -> is_protein` map is built from `metadata_df` itself. This
    assumes all contacted pn_unit iids appearing in `contacts_col` also have a
    row in `metadata_df` (same (pdb_id, assembly_id) partitioning); unknown iids
    default to `is_protein=False`, matching the v3 step3 semantic.
    """
    required = [contacts_col, protein_col, iid_col]
    missing = [c for c in required if c not in metadata_df.columns]
    if missing:
        raise KeyError(
            f"derive_context_columns requires columns {missing} on metadata_df."
        )

    # (pdb_id, assembly_id, iid) -> is_protein: mirrors step3's keying so that the
    # same iid string in a different assembly isn't wrongly resolved.
    has_assembly_key = (
        "pdb_id" in metadata_df.columns and "assembly_id" in metadata_df.columns
    )
    if has_assembly_key:
        keys = list(
            zip(
                metadata_df["pdb_id"],
                metadata_df["assembly_id"].astype(str),
                metadata_df[iid_col],
            )
        )
        pn_unit_is_protein_map = dict(zip(keys, metadata_df[protein_col].astype(bool)))
    else:
        pn_unit_is_protein_map = dict(
            zip(metadata_df[iid_col], metadata_df[protein_col].astype(bool))
        )

    def _parse_contacts(raw):
        if raw is None:
            return []
        # pandas stores NaN as float; lists/dicts don't need isna scrutiny
        if isinstance(raw, float) and pd.isna(raw):
            return []
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return []
        # Already list-like (list, tuple, numpy array of dicts)
        return list(raw)

    def _per_row(row):
        self_iid = row[iid_col]
        context = [self_iid]
        n_protein = 0

        contacts = _parse_contacts(row[contacts_col])
        for contact in contacts:
            if not isinstance(contact, dict):
                continue
            cid = contact.get("pn_unit_iid")
            md = contact.get("min_distance")
            if cid is None or md is None:
                continue
            if md > context_distance:
                continue
            context.append(cid)
            if has_assembly_key:
                key = (row["pdb_id"], str(row["assembly_id"]), cid)
            else:
                key = cid
            if pn_unit_is_protein_map.get(key, False):
                n_protein += 1

        return context, n_protein

    results = metadata_df.apply(_per_row, axis=1, result_type="expand")

    out = metadata_df.copy()
    out["q_pn_unit_context_group_iids"] = results[0]
    out["num_contacting_protein_chains"] = results[1].astype("int32")
    return out


def apply_context_group_chain_type_whitelist(
    metadata_df: pd.DataFrame,
    include_pmsm: bool = True,
    include_pmm: bool = True,
    include_branched: bool = True,
    include_nuc: bool = True,
    include_peptide: bool = True,
    iid_col: str = "q_pn_unit_iid",
    context_col: str = "q_pn_unit_context_group_iids",
) -> pd.DataFrame:
    """Restrict `q_pn_unit_context_group_iids` to a whitelist of chain types.

    A contacted pn_unit is kept in the context group iff its own metadata row
    satisfies at least one of the enabled whitelist flags:
      - `q_pn_unit_is_physically_meaningful_small_molecule`  (include_pmsm)
      - `q_pn_unit_is_physically_meaningful_metal`           (include_pmm)
      - `q_pn_unit_type == ChainType.BRANCHED`               (include_branched)
      - `q_pn_unit_is_nuc`                                    (include_nuc, DNA/RNA/hybrid)
      - `q_pn_unit_is_peptide`                                (include_peptide)

    The row's own iid (self) is always preserved to match the invariant established
    by `derive_context_columns` (each chain's context group contains itself).

    Missing flag columns are treated as all-False for their contribution (i.e.,
    disabling that slice of the whitelist), so this filter is safe to call on
    parquets that don't ship every optional flag. The caller should compute
    PMSM / PMM runtime columns before invoking this helper when those toggles
    are on.
    """
    required = [iid_col, context_col, "pdb_id", "assembly_id"]
    missing = [c for c in required if c not in metadata_df.columns]
    if missing:
        raise KeyError(
            f"apply_context_group_chain_type_whitelist requires columns {missing}."
        )

    # Row-wise "allowed as context group member" mask — union of enabled slices.
    allowed = pd.Series(False, index=metadata_df.index)

    def _or_col(mask: pd.Series, col: str, enabled: bool) -> pd.Series:
        if not enabled:
            return mask
        if col not in metadata_df.columns:
            logger.warning(
                f"apply_context_group_chain_type_whitelist: column '{col}' not present; "
                f"treating that slice as empty."
            )
            return mask
        return mask | metadata_df[col].fillna(False).astype(bool)

    allowed = _or_col(allowed, "q_pn_unit_is_physically_meaningful_small_molecule", include_pmsm)
    allowed = _or_col(allowed, "q_pn_unit_is_physically_meaningful_metal", include_pmm)
    allowed = _or_col(allowed, "q_pn_unit_is_nuc", include_nuc)
    allowed = _or_col(allowed, "q_pn_unit_is_peptide", include_peptide)
    if include_branched:
        if "q_pn_unit_type" in metadata_df.columns:
            allowed = allowed | (metadata_df["q_pn_unit_type"] == aw_enums.ChainType.BRANCHED.value)
        else:
            logger.warning(
                "apply_context_group_chain_type_whitelist: 'q_pn_unit_type' not present; "
                "branched slice disabled."
            )

    keys = list(
        zip(
            metadata_df["pdb_id"],
            metadata_df["assembly_id"].astype(str),
            metadata_df[iid_col],
        )
    )
    iid_allowed_map = dict(zip(keys, allowed.astype(bool)))

    def _filter_row(row):
        self_iid = row[iid_col]
        iids = row[context_col]
        if iids is None:
            return None
        pdb_id = row["pdb_id"]
        asm = str(row["assembly_id"])
        kept = []
        for iid in iids:
            if iid == self_iid or iid_allowed_map.get((pdb_id, asm, iid), False):
                kept.append(iid)
        return kept

    before_lens = metadata_df[context_col].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    out = metadata_df.copy()
    out[context_col] = metadata_df.apply(_filter_row, axis=1)
    after_lens = out[context_col].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    if len(before_lens):
        logger.info(
            f"Context group chain-type whitelist applied "
            f"(pmsm={include_pmsm}, pmm={include_pmm}, branched={include_branched}, "
            f"nuc={include_nuc}, peptide={include_peptide}): "
            f"avg length {before_lens.mean():.3f} -> {after_lens.mean():.3f} "
            f"(rows with any contact: {(before_lens > 1).sum():,} -> "
            f"{(after_lens > 1).sum():,})"
        )
    return out
