import json
import argparse
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ALL_PN_UNIT_IIDS_COL = "all_pn_unit_iids_after_processing"
Q_CLOSE_IIDS_COL = "q_pn_unit_close_pn_unit_iids"
Q_CONTACTING_COL = "q_pn_unit_contacting_pn_unit_iids"


def _json_loads_maybe(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, str):
        return json.loads(x)
    return x


def _json_dumps(x: Any) -> str:
    # compact + stable-ish formatting
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False)


def _filter_iids(iids: Any, excluded_iids: set[str]) -> Any:
    iids = _json_loads_maybe(iids)
    if iids is None:
        return _json_dumps([])
    if not isinstance(iids, list):
        raise TypeError(f"Expected list[str] (or json string), got {type(iids)}")
    return _json_dumps([iid for iid in iids if iid not in excluded_iids])


def _filter_contacting(contacting: Any, excluded_iids: set[str]) -> Any:
    contacting = _json_loads_maybe(contacting)
    if contacting is None:
        return _json_dumps([])
    if not isinstance(contacting, list):
        raise TypeError(f"Expected list[dict] (or json string), got {type(contacting)}")
    kept = []
    for d in contacting:
        # expected shape: {"pn_unit_iid": "...", ...}
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict in contacting list, got {type(d)}")
        iid = d.get("pn_unit_iid")
        if iid is None or iid not in excluded_iids:
            kept.append(d)
    return _json_dumps(kept)


def _count_excluded_hits_in_row(row: pd.Series, excluded_iids: set[str]) -> tuple[int, int, int]:
    all_iids = _json_loads_maybe(row.get(ALL_PN_UNIT_IIDS_COL, "[]")) or []
    close_iids = _json_loads_maybe(row.get(Q_CLOSE_IIDS_COL, "[]")) or []
    contacting = _json_loads_maybe(row.get(Q_CONTACTING_COL, "[]")) or []
    n_all = sum(iid in excluded_iids for iid in all_iids) if isinstance(all_iids, list) else 0
    n_close = sum(iid in excluded_iids for iid in close_iids) if isinstance(close_iids, list) else 0
    n_contact = 0
    if isinstance(contacting, list):
        for d in contacting:
            if isinstance(d, dict) and d.get("pn_unit_iid") in excluded_iids:
                n_contact += 1
    return n_all, n_close, n_contact


def remove_excluded_iids_from_neighbor_cols(
    df: pd.DataFrame,
    exclusion_ccd_codes: Iterable[str],
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Apply row-level exclusion (based on `q_pn_unit_non_polymer_res_names`) and then,
    within the same `pdb_id`, remove excluded `pn_unit_iid`s from:
    - `all_pn_unit_iids_after_processing` (JSON string of list[str])
    - `q_pn_unit_close_pn_unit_iids` (JSON string of list[str])
    - `q_pn_unit_contacting_pn_unit_iids` (JSON string of list[dict])
    """
    exclusion_ccd_codes = list(exclusion_ccd_codes)

    required_cols = ["pdb_id", "q_pn_unit_iid", "q_pn_unit_non_polymer_res_names"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df if inplace else df.copy()

    # 1) pdb_id별로 제외 대상 iid set 생성 (row-level exclusion과 동일 기준)
    excluded_rows = work["q_pn_unit_non_polymer_res_names"].isin(exclusion_ccd_codes)
    excluded_iids_by_pdb = (
        work.loc[excluded_rows, ["pdb_id", "q_pn_unit_iid"]]
        .groupby("pdb_id")["q_pn_unit_iid"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )

    # 2) row-level exclusion 적용
    work = work.loc[~excluded_rows].copy()
    work.reset_index(drop=True, inplace=True)

    # 3) neighbor columns에서 제외 iid 제거
    if ALL_PN_UNIT_IIDS_COL in work.columns:
        work[ALL_PN_UNIT_IIDS_COL] = work.apply(
            lambda r: _filter_iids(r[ALL_PN_UNIT_IIDS_COL], excluded_iids_by_pdb.get(r["pdb_id"], set())),
            axis=1,
        )
    if Q_CLOSE_IIDS_COL in work.columns:
        work[Q_CLOSE_IIDS_COL] = work.apply(
            lambda r: _filter_iids(r[Q_CLOSE_IIDS_COL], excluded_iids_by_pdb.get(r["pdb_id"], set())),
            axis=1,
        )
    if Q_CONTACTING_COL in work.columns:
        work[Q_CONTACTING_COL] = work.apply(
            lambda r: _filter_contacting(r[Q_CONTACTING_COL], excluded_iids_by_pdb.get(r["pdb_id"], set())),
            axis=1,
        )

    return work


def get_deleted_chain_df(
    original_df: pd.DataFrame,
    exclusion_ccd_codes: Iterable[str],
) -> pd.DataFrame:
    """
    Return the list of chains (rows) removed by the row-level exclusion
    (e.g., columns like pdb_id, q_pn_unit_iid, q_pn_unit_id, q_pn_unit_non_polymer_res_names).
    """
    exclusion_ccd_codes = list(exclusion_ccd_codes)
    mask = original_df["q_pn_unit_non_polymer_res_names"].isin(exclusion_ccd_codes)
    deleted = original_df.loc[mask].copy()

    # Keep a compact, human-friendly column subset (only if present).
    preferred = [
        "pdb_id",
        "example_id",
        "q_pn_unit_iid",
        "q_pn_unit_id",
        "q_pn_unit_non_polymer_res_names",
        "q_pn_unit_is_polymer",
        "q_pn_unit_type",
        "q_pn_unit_asym_id",
        "q_pn_unit_auth_asym_id",
        "q_pn_unit_entity_id",
        "q_pn_unit_processed_entity_id",
    ]
    cols = [c for c in preferred if c in deleted.columns]
    if cols:
        deleted = deleted[cols]
    return deleted


def _parse_exclusion_ccd_codes(args: argparse.Namespace) -> list[str]:
    """
    Priority:
      1) --exclude-ccd-codes-txt
      2) --exclude-ccd-codes (comma-separated or repeated)
      3) default fallback
    """
    if args.exclude_ccd_codes_txt:
        p = Path(args.exclude_ccd_codes_txt)
        if not p.exists():
            raise FileNotFoundError(f"exclude ccd code txt not found: {p}")
        codes: list[str] = []
        for line in p.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            codes.append(s)
        if not codes:
            raise ValueError(f"No CCD codes found in txt: {p}")
        return codes

    if args.exclude_ccd_codes:
        # Support repeated flags and comma-separated values.
        out: list[str] = []
        for item in args.exclude_ccd_codes:
            for s in str(item).split(","):
                s = s.strip()
                if s:
                    out.append(s)
        if out:
            return out

    # Default fallback
    return ["NA", "K", "CL", "BR"]


def _save_df(df: pd.DataFrame, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    else:
        # Default: parquet
        df.to_parquet(out, index=False)


def validate_no_excluded_iids_in_neighbor_cols(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    exclusion_ccd_codes: Iterable[str],
) -> None:
    exclusion_ccd_codes = list(exclusion_ccd_codes)
    excluded_rows = original_df["q_pn_unit_non_polymer_res_names"].isin(exclusion_ccd_codes)
    excluded_iids_by_pdb = (
        original_df.loc[excluded_rows, ["pdb_id", "q_pn_unit_iid"]]
        .groupby("pdb_id")["q_pn_unit_iid"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )

    # Count hits after cleaning
    totals = [0, 0, 0]
    per_col_any = [0, 0, 0]
    for _, row in cleaned_df.iterrows():
        ex = excluded_iids_by_pdb.get(row["pdb_id"], set())
        n_all, n_close, n_contact = _count_excluded_hits_in_row(row, ex)
        totals[0] += n_all
        totals[1] += n_close
        totals[2] += n_contact
        per_col_any[0] += int(n_all > 0)
        per_col_any[1] += int(n_close > 0)
        per_col_any[2] += int(n_contact > 0)

    print("Validation (after cleaning):")
    print(f"- rows: {len(cleaned_df):,}")
    print(f"- excluded-hit total counts: all={totals[0]:,}, close={totals[1]:,}, contacting={totals[2]:,}")
    print(f"- rows_with_any_hit: all={per_col_any[0]:,}, close={per_col_any[1]:,}, contacting={per_col_any[2]:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude-ccd-codes",
        action="append",
        default='NA,K,CL,BR',
        help="CCD codes to exclude. Example: --exclude-ccd-codes NA --exclude-ccd-codes CL or --exclude-ccd-codes NA,K,CL",
    )
    parser.add_argument(
        "--exclude-ccd-codes-txt",
        default=None,
        help="Path to a txt file with one CCD code per line (supports '#' comments).",
    )
    parser.add_argument(
        "--input-parquet",
        default="/home/possu/jinho/datasets/atomworks_pdb_full_v5/debug_metadata_seq_clustered_04.parquet",
        help="Input parquet path.",
    )
    parser.add_argument(
        "--output-parquet",
        default="/home/possu/jinho/datasets/atomworks_pdb_full_v5/debug_metadata_seq_clustered_04_lmpnn.parquet",
        help="Output parquet path (if suffix is .csv, writes CSV).",
    )
    parser.add_argument(
        "--deleted-chains-out",
        default="/home/possu/jinho/datasets/atomworks_pdb_full_v5/debug_metadata_seq_clustered_04_deleted_chains.parquet",
        help="Output path for the deleted-chain list (parquet/csv).",
    )
    args = parser.parse_args()

    exclusion_ccd_codes = _parse_exclusion_ccd_codes(args)
    summary_lines: list[str] = []
    summary_lines.append(f"Exclusion CCD codes ({len(exclusion_ccd_codes)}): {exclusion_ccd_codes}")
    summary_lines.append(f"Input : {args.input_parquet}")
    summary_lines.append(f"Output: {args.output_parquet}")
    summary_lines.append(f"Deleted chains out: {args.deleted_chains_out}")
    print("\n".join(summary_lines))

    input_path = Path(args.input_parquet)
    summary_path = input_path.parent / f"{input_path.stem}_deleted_chains_summary.txt"

    df = pd.read_parquet(input_path)
    summary_lines.append(f"Input rows: {len(df):,}")

    deleted_df = get_deleted_chain_df(df, exclusion_ccd_codes=exclusion_ccd_codes)
    _save_df(deleted_df, args.deleted_chains_out)
    summary_lines.append(f"Deleted chain rows (row-level exclusion): {len(deleted_df):,}")
    print(f"Saved deleted chains list: {args.deleted_chains_out} (rows={len(deleted_df):,})")

    cleaned = remove_excluded_iids_from_neighbor_cols(df, exclusion_ccd_codes=exclusion_ccd_codes)
    summary_lines.append(f"Output rows: {len(cleaned):,}")

    # capture validation stats for summary
    excluded_rows = df["q_pn_unit_non_polymer_res_names"].isin(list(exclusion_ccd_codes))
    excluded_iids_by_pdb = (
        df.loc[excluded_rows, ["pdb_id", "q_pn_unit_iid"]]
        .groupby("pdb_id")["q_pn_unit_iid"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )
    totals = [0, 0, 0]
    per_col_any = [0, 0, 0]
    for _, row in cleaned.iterrows():
        ex = excluded_iids_by_pdb.get(row["pdb_id"], set())
        n_all, n_close, n_contact = _count_excluded_hits_in_row(row, ex)
        totals[0] += n_all
        totals[1] += n_close
        totals[2] += n_contact
        per_col_any[0] += int(n_all > 0)
        per_col_any[1] += int(n_close > 0)
        per_col_any[2] += int(n_contact > 0)

    print("Validation (after cleaning):")
    print(f"- rows: {len(cleaned):,}")
    print(f"- excluded-hit total counts: all={totals[0]:,}, close={totals[1]:,}, contacting={totals[2]:,}")
    print(f"- rows_with_any_hit: all={per_col_any[0]:,}, close={per_col_any[1]:,}, contacting={per_col_any[2]:,}")

    summary_lines.append("Validation (after cleaning):")
    summary_lines.append(f"- excluded-hit total counts: all={totals[0]:,}, close={totals[1]:,}, contacting={totals[2]:,}")
    summary_lines.append(f"- rows_with_any_hit: all={per_col_any[0]:,}, close={per_col_any[1]:,}, contacting={per_col_any[2]:,}")

    _save_df(cleaned, args.output_parquet)
    print(f"Saved filtered metadata: {args.output_parquet} (rows={len(cleaned):,})")
    summary_lines.append(f"Saved filtered metadata: {args.output_parquet}")

    # write summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Saved summary: {summary_path}")