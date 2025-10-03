"""
ECFP4/Tanimoto 기반 소분자(리간드) 클러스터링과 금속 클러스터링을 추가한 버전.

기존 동작:
- 폴리머(단백질/핵산)는 기존 로직 유지 (단백질: mmseqs, 짧은 펩타이드/핵산: 유니크 ID)

추가 동작:
- 금속: CCD 코드(=원소 기호) 단위로 클러스터링 (동일 금속은 같은 클러스터)
- 소분자: ECFP4 임베딩 기반 Tanimoto 유사도(기본 0.5)로 연결 요소 클러스터링
  - SMILES 컬럼이 없거나 RDKit 미설치 시에는 기존의 `q_pn_unit_non_polymer_res_names` 기반 유니크 그룹으로 폴백

Hydra 설정:
- 기존 `cluster_sequences` 설정을 재사용합니다.
- 다음 선택적 설정을 지원합니다(없으면 기본값 사용):
  - `small_mol_tanimoto_threshold` (float, 기본 0.5)
  - `small_mol_ecfp_nbits` (int, 기본 2048)
  - `small_mol_smiles_column` (str, 기본적으로 자동 탐색)
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import atomworks.enums as aw_enums
import hydra
import numpy as np
import pandas as pd
from atomworks.ml.preprocessing.constants import PEPTIDE_MAX_RESIDUES
from atomworks.ml.utils.misc import hash_sequence
from omegaconf import DictConfig
from tqdm import tqdm

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False


# 흔히 등장하는 금속 이온 HET 코드(원소 기호 기반)
METAL_ELEMENT_CODES: Set[str] = {
    "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA",
    "SC", "Y", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
    "GA", "GE", "AS", "SE", "BR", "KR",  # 비금속/준금속 포함 가능, 필요시 조정
    "RB", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD",
    "IN", "SN", "SB", "TE", "I",  # 할로겐/준금속 포함, 금속만으로 좁히고 싶으면 정제 가능
    "PT", "AU", "HG", "TL", "PB", "BI",
    "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB", "DY", "HO",
    "ER", "TM", "YB", "LU", "HF", "TA", "W", "RE", "OS", "IR"
}

# SMILES 컬럼 후보 (우선순위 순서)
SMILES_CANDIDATE_COLUMNS: List[str] = [
    "q_pn_unit_ligand_smiles",
    "q_pn_unit_non_polymer_smiles",
    "q_pn_unit_smiles",
    "ligand_smiles",
    "smiles",
]

DEFAULT_ECFP_NBITS = 2048
DEFAULT_TANIMOTO_THRESHOLD = 0.5


def find_smiles_column(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    for col in SMILES_CANDIDATE_COLUMNS:
        if col in df.columns:
            return col
    return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not RDKit_AVAILABLE:
        return None
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def compute_ecfp4_fingerprints(smiles_list: List[str], n_bits: int) -> List[Tuple[str, Optional[object]]]:
    """
    입력: canonical SMILES 리스트
    출력: (smiles, fp) 리스트. mol 파싱 실패 시 fp=None
    """
    if not RDKit_AVAILABLE:
        return [(s, None) for s in smiles_list]
    fps: List[Tuple[str, Optional[object]]] = []
    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                fps.append((s, None))
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
                fps.append((s, fp))
        except Exception:
            fps.append((s, None))
    return fps


def cluster_fingerprints_connected_components(fps: List[Tuple[str, object]], threshold: float) -> Dict[str, int]:
    """
    간단한 연결 요소 클러스터링.
    - fps: (smiles, rdkit-bitvect) 목록. None fp는 단독 클러스터로 취급.
    - threshold 이상이면 엣지 생성 -> 연결 요소가 같은 클러스터.
    반환: smiles -> local cluster id (0..K-1)
    """
    # None fp는 먼저 단독 클러스터로 배정
    smiles = [s for s, _ in fps]
    rd_fps = [fp for _, fp in fps]
    n = len(fps)

    # 모든 None에 대해 고유 클러스터를 먼저 지정
    cluster_ids: Dict[int, int] = {}
    current_id = 0
    for i, fp in enumerate(rd_fps):
        if fp is None:
            cluster_ids[i] = current_id
            current_id += 1

    # 유효 fp 인덱스만으로 그래프 구성
    valid_idx = [i for i, fp in enumerate(rd_fps) if fp is not None]
    m = len(valid_idx)
    if m > 0:
        # 인접 리스트
        adj: List[Set[int]] = [set() for _ in range(n)]
        for ix, i in enumerate(valid_idx):
            base_fp = rd_fps[i]
            others = [rd_fps[j] for j in valid_idx[ix + 1 :]]
            if RDKit_AVAILABLE and base_fp is not None and len(others) > 0:
                sims = DataStructs.BulkTanimotoSimilarity(base_fp, others)
            else:
                sims = []
            for off, sim in enumerate(sims, start=ix + 1):
                j = valid_idx[off]
                if sim >= threshold:
                    adj[i].add(j)
                    adj[j].add(i)

        # 연결 요소 탐색(BFS)
        for i in valid_idx:
            if i in cluster_ids:
                continue
            queue = [i]
            cluster_ids[i] = current_id
            while queue:
                u = queue.pop()
                for v in adj[u]:
                    if v not in cluster_ids:
                        cluster_ids[v] = current_id
                        queue.append(v)
            current_id += 1

    # smiles -> local cluster id 매핑 생성
    smiles_to_local: Dict[str, int] = {}
    for i, s in enumerate(smiles):
        smiles_to_local[s] = cluster_ids[i]
    return smiles_to_local


def extract_metal_code(res_names: Optional[str]) -> Optional[str]:
    """
    `q_pn_unit_non_polymer_res_names`에서 금속 단일 HET 코드인지 판정.
    - 단일 토큰이고 METAL_ELEMENT_CODES에 있으면 금속으로 간주.
    - 복합/여러 토큰이면 None
    """
    if not isinstance(res_names, str) or res_names.strip() == "":
        return None
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", res_names.upper()) if t]
    if len(tokens) == 1 and tokens[0] in METAL_ELEMENT_CODES:
        return tokens[0]
    return None


@hydra.main(
    config_path="../../../configs/data/preprocessing/atomworks",
    config_name="cluster_sequences",
    version_base="1.3.2",
)
def main(cfg: DictConfig) -> None:
    """
    메타데이터 파일을 읽어 폴리머/소분자/금속을 통합 클러스터링하고,
    `q_pn_unit_cluster_id`를 추가 저장합니다.
    """
    df = pd.read_parquet(f"{cfg.pdb_path}/metadata.parquet")
    clustering_dir = Path(cfg.pdb_path) / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    # 파라미터
    tanimoto_threshold: float = float(getattr(cfg, "small_mol_tanimoto_threshold", DEFAULT_TANIMOTO_THRESHOLD))
    ecfp_nbits: int = int(getattr(cfg, "small_mol_ecfp_nbits", DEFAULT_ECFP_NBITS))
    preferred_smiles_col: Optional[str] = getattr(cfg, "small_mol_smiles_column", None)

    # -------------------------
    # 1) 폴리머 분류 및 군집화 입력 준비
    # -------------------------
    proteins: Set[str] = set()
    shorts: Set[str] = set()
    nucleic_acids: Set[str] = set()

    for _, row in tqdm(df.iterrows(), desc="Sorting sequences by type", total=len(df)):
        chain_type = row["q_pn_unit_type"]
        if chain_type in aw_enums.ChainTypeInfo.PROTEINS:
            seq = row["q_pn_unit_processed_entity_canonical_sequence"]
            if isinstance(seq, str) and len(seq) <= PEPTIDE_MAX_RESIDUES:
                shorts.add(seq)
            else:
                proteins.add(seq)
        elif chain_type in aw_enums.ChainTypeInfo.NUCLEIC_ACIDS:
            nucleic_acids.add(row["q_pn_unit_processed_entity_canonical_sequence"])

    # -------------------------
    # 2) 단백질: mmseqs 클러스터링
    # -------------------------
    clustering_label_map: Dict[str, str] = {}

    if len(proteins) > 0:
        fasta_records = [f">{hash_sequence(seq)}\n{seq}" for seq in proteins]
        with (clustering_dir / "proteins.fasta").open("w") as f:
            f.write("\n".join(fasta_records))

        subprocess.run(
            f"{os.environ['SOFTWARE_PATH']}/mmseqs/bin/mmseqs easy-cluster {clustering_dir / 'proteins.fasta'} {clustering_dir / 'clust_prot'} {clustering_dir / 'tmp'} --min-seq-id 0.4",
            shell=True,
            check=True,
        )

        clustering_path = clustering_dir / "clust_prot_cluster.tsv"
        protein_data = pd.read_csv(clustering_path, sep="\t", header=None)
        clusters = protein_data[0]
        items = protein_data[1]
        for cluster_label, item_id in zip(clusters, items):
            # 키: 폴리머 해시
            key = f"poly::{item_id}"
            val = f"prot::{cluster_label}"
            clustering_label_map[key] = val

    # 짧은 펩타이드/핵산: 유니크 ID
    for short in shorts:
        sid = hash_sequence(short)
        clustering_label_map[f"poly::{sid}"] = f"short::{sid}"

    for nucl in nucleic_acids:
        nid = hash_sequence(nucl)
        clustering_label_map[f"poly::{nid}"] = f"nucl::{nid}"

    # -------------------------
    # 3) 금속/소분자 처리 (비폴리머)
    # -------------------------
    is_polymer = df["q_pn_unit_is_polymer"].astype(bool)
    non_poly_df = df.loc[~is_polymer].copy()

    # 금속 판정
    non_poly_df["_metal_code"] = non_poly_df["q_pn_unit_non_polymer_res_names"].apply(extract_metal_code)
    metal_codes: Set[str] = set(non_poly_df["_metal_code"].dropna().astype(str).tolist())

    for code in sorted(metal_codes):
        clustering_label_map[f"metal::{code}"] = f"metal::{code}"

    # 소분자 SMILES 컬럼 탐색 및 정규화
    smiles_col = find_smiles_column(non_poly_df, preferred=preferred_smiles_col)

    small_mol_smiles: List[str] = []
    if smiles_col is not None:
        # 금속이 아닌 행만 대상으로 SMILES 사용
        small_df = non_poly_df.loc[non_poly_df["_metal_code"].isna() & non_poly_df[smiles_col].notna(), [smiles_col]].copy()
        # canonicalization
        if RDKit_AVAILABLE:
            small_df["_csmiles"] = small_df[smiles_col].apply(canonicalize_smiles)
        else:
            small_df["_csmiles"] = None
        small_mol_smiles = [s for s in small_df["_csmiles"].dropna().astype(str).unique().tolist()]

    # ECFP4 + 연결요소 클러스터링
    smiles_to_local_cluster: Dict[str, int] = {}
    if RDKit_AVAILABLE and len(small_mol_smiles) > 0:
        fps = compute_ecfp4_fingerprints(small_mol_smiles, n_bits=ecfp_nbits)
        smiles_to_local_cluster = cluster_fingerprints_connected_components(fps, threshold=tanimoto_threshold)

        # local id -> 전역 라벨 문자열 구성
        # 같은 local id를 갖는 모든 smiles에 동일 전역 라벨을 부여
        inv: Dict[int, List[str]] = {}
        for s, cid in smiles_to_local_cluster.items():
            inv.setdefault(cid, []).append(s)
        for cid, members in inv.items():
            label = f"smcl::{cid}"
            for s in members:
                key = f"smiles::{hash_sequence(s)}"
                clustering_label_map[key] = label
    else:
        # RDKit 미가용/SMILES 부재: 폴백 - res_names 기반 유니크 그룹
        fallback_nonmet = non_poly_df.loc[non_poly_df["_metal_code"].isna(), "q_pn_unit_non_polymer_res_names"].dropna().astype(str).unique().tolist()
        for res_names in fallback_nonmet:
            key = f"nonpoly::{hash_sequence(res_names)}"
            clustering_label_map[key] = key  # 자체 라벨

    # -------------------------
    # 4) 전역 정수 클러스터 ID 부여
    # -------------------------
    unique_labels = sorted(set(clustering_label_map.values()))
    label_to_int: Dict[str, int] = {lab: i for i, lab in enumerate(unique_labels)}
    clustering: Dict[str, int] = {k: label_to_int[v] for k, v in clustering_label_map.items()}

    # -------------------------
    # 5) 각 행별 클러스터 키 생성 및 매핑
    # -------------------------
    smiles_col_global = find_smiles_column(df, preferred=preferred_smiles_col)

    def row_to_cluster_key(row: pd.Series) -> str:
        if bool(row["q_pn_unit_is_polymer"]):
            poly_hash = row["q_pn_unit_processed_entity_canonical_sequence_hash"]
            return f"poly::{poly_hash}"
        # 비폴리머
        res_names = row.get("q_pn_unit_non_polymer_res_names", None)
        metal_code = extract_metal_code(res_names)
        if metal_code is not None:
            return f"metal::{metal_code}"
        if smiles_col_global is not None:
            s = row.get(smiles_col_global, None)
            csmiles = canonicalize_smiles(s) if RDKit_AVAILABLE else None
            if csmiles:
                return f"smiles::{hash_sequence(csmiles)}"
        # 폴백: res_names 기반
        rn = res_names if isinstance(res_names, str) else ""
        return f"nonpoly::{hash_sequence(rn)}"

    df["_cluster_key"] = df.apply(row_to_cluster_key, axis=1)
    df["q_pn_unit_cluster_id"] = df["_cluster_key"].map(clustering)

    # 누락 경고 및 채움
    if df["q_pn_unit_cluster_id"].isna().any():
        print(
            f"WARNING: {df['q_pn_unit_cluster_id'].isna().sum()} missing values in q_pn_unit_cluster_id. Filling with -1."
        )
    df["q_pn_unit_cluster_id"] = df["q_pn_unit_cluster_id"].fillna(-1).astype(np.int32)

    # 저장
    with (clustering_dir / "clustering.json").open("w") as handle:
        json.dump(clustering, handle)

    df.to_parquet(f"{cfg.pdb_path}/metadata_clustered.parquet")
    print(f"Saved clustered metadata to {cfg.pdb_path}/metadata_clustered.parquet")


if __name__ == "__main__":
    main()



