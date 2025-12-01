import pandas as pd
from atomworks.enums import ChainTypeInfo, ChainType
from atomworks.ml.preprocessing.constants import ENTRIES_TO_EXCLUDE_FOR_PRE_PROCESSING
from atomworks.constants import METAL_ELEMENTS
from atomworks.ml.preprocessing.constants import PEPTIDE_MAX_RESIDUES, NUCLEIC_ACID_LIGANDS_MAX_RESIDUES
import json, ast, re
from collections import defaultdict, deque
import argparse
from multiprocessing import Pool, cpu_count

def metadata_ligand_chain_clustering(args: argparse.Namespace):
    '''
    Followed the definition of ChainTypeInfo in atomworks/src/atomworks/enums.py 
    Note: 
    
    Didn't consider ChainType.OTHER_POLYMER. There are only 3ok2, 3ok4 in pdb dataset.
    Consider only polypeptide-L chain as a protein
    
    
    '''

    input_parquet_path = args.input_parquet_path
    output_dir_path = args.output_dir_path
    debug = args.debug
    # ensure proper types for CLI-provided strings
    try:
        debug_num_pdb_ids = int(args.debug_num_pdb_ids)
    except Exception:
        debug_num_pdb_ids = 1000
    try:
        args.nucleic_acid_dist_threshold = float(args.nucleic_acid_dist_threshold)
    except Exception:
        args.nucleic_acid_dist_threshold = 4.0
    try:
        args.ligand_cluster_dist_threshold = float(getattr(args, 'ligand_cluster_dist_threshold', 5.0))
    except Exception:
        args.ligand_cluster_dist_threshold = 5.0

    # Load the original metadata parquet
    atomworks_parquet = pd.read_parquet(input_parquet_path)

    ### Proteins, only consider polypeptide-L chain as a protein
    protein_chain_type = ChainType.POLYPEPTIDE_L
    
    ### Peptide-like short-polymer ligands
    peptide_chain_type = [ChainType.POLYPEPTIDE_D, ChainType.POLYPEPTIDE_L, ChainType.CYCLIC_PSEUDO_PEPTIDE, ChainType.PEPTIDE_NUCLEIC_ACID]
        
    ### Nucleic acids
    DNA_chain_type_values = [ChainType.DNA.value]
    RNA_chain_type_values = [ChainType.RNA.value]
    RNA_DNA_hybrid_chain_type_values = [ChainType.DNA_RNA_HYBRID.value]

    ### Ligands
    ligand_chain_types = ChainTypeInfo.NON_POLYMERS
    ligand_chain_type_values = [chain_type.value for chain_type in ligand_chain_types]

    # add "is_protein" & "is_peptide" column, following the definition in Atomworks
    atomworks_parquet["q_pn_unit_is_protein"] = (atomworks_parquet["q_pn_unit_type"] == protein_chain_type) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] >= PEPTIDE_MAX_RESIDUES)
    atomworks_parquet["q_pn_unit_is_peptide"] = (atomworks_parquet["q_pn_unit_type"].isin(peptide_chain_type)) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] < PEPTIDE_MAX_RESIDUES)
    
    # Take only pdb_id with at least one protein chain        
    mask = atomworks_parquet.groupby('pdb_id')['q_pn_unit_is_protein'].transform('any')
    atomworks_parquet = atomworks_parquet.loc[mask].reset_index(drop=True)
    
    # Small molecule ligands & small molecule - metal complexes
    atomworks_parquet["q_pn_unit_is_small_molecule"] = (atomworks_parquet["q_pn_unit_type"].isin(ligand_chain_type_values)) & (atomworks_parquet["q_pn_unit_is_metal"] == False)
            
    # nucleotides
    atomworks_parquet["q_pn_unit_is_DNA"] = atomworks_parquet["q_pn_unit_type"].isin(DNA_chain_type_values)
    atomworks_parquet["q_pn_unit_is_RNA"] = atomworks_parquet["q_pn_unit_type"].isin(RNA_chain_type_values)
    atomworks_parquet["q_pn_unit_is_RNA_DNA_hybrid"] = atomworks_parquet["q_pn_unit_type"].isin(RNA_DNA_hybrid_chain_type_values)
    
    if debug:
        debug_pdb_ids = atomworks_parquet['pdb_id'].unique()[:debug_num_pdb_ids]
        atomworks_parquet = atomworks_parquet[atomworks_parquet['pdb_id'].isin(debug_pdb_ids)]                

    ### Clustering non-protein chains
    def _split_components(iid):
        if not isinstance(iid, str):
            iid = str(iid)
        if ',' in iid:
            return [tok.strip() for tok in iid.split(',') if tok.strip()]
        return [iid.strip()] if iid else []

    def _parse_contacts(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        if isinstance(val, list):
            items = val
        else:
            s = val.decode('utf-8', 'ignore') if isinstance(val, (bytes, bytearray)) else str(val)
            s = s.strip()
            if not s:
                return []
            try:
                items = json.loads(s)
            except Exception:
                try:
                    items = ast.literal_eval(s)
                except Exception:
                    return []
        out = []
        for item in items:
            if isinstance(item, dict) and 'pn_unit_iid' in item and item['pn_unit_iid']:
                out.append(str(item['pn_unit_iid']))
        return out

    def _parse_contacts_with_distance(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        if isinstance(val, list):
            items = val
        else:
            s = val.decode('utf-8', 'ignore') if isinstance(val, (bytes, bytearray)) else str(val)
            s = s.strip()
            if not s:
                return []
            try:
                items = json.loads(s)
            except Exception:
                try:
                    items = ast.literal_eval(s)
                except Exception:
                    return []
        out = []
        for item in items:
            if isinstance(item, dict) and 'pn_unit_iid' in item and item['pn_unit_iid']:
                md = item.get('min_distance', None)
                try:
                    md = float(md) if md is not None else None
                except Exception:
                    md = None
                out.append((str(item['pn_unit_iid']), md))
        return out

    def _natural_key(s):
        return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]

    def _join_sorted(ids):
        ids = list(ids) if ids is not None else []
        if not ids:
            return ""
        return ", ".join(sorted(ids, key=_natural_key))
        
    def _assign_nucleic_acid_chain_clusters_per_pdb(g):
        # Only DNA/RNA/RNA-DNA hybrid chains are considered
        na_mask = (g['q_pn_unit_is_DNA'] | g['q_pn_unit_is_RNA'] | g['q_pn_unit_is_RNA_DNA_hybrid'])
        if not na_mask.any():
            return pd.Series(index=g.index, dtype=object)

        sub = g.loc[na_mask, ['q_pn_unit_iid', 'q_pn_unit_contacting_pn_unit_iids']].copy()

        # Map chain id to atomic components (comma-separated)
        rid_to_comps = {str(rid): tuple(_split_components(str(rid))) for rid in sub['q_pn_unit_iid'].astype(str)}
        atomic_nodes = set(c for comps in rid_to_comps.values() for c in comps)

        # Collect list of (target chain, min_distance) reported by each chain
        rid_contacts_pairs = {}
        for rid_raw, ssub in sub.groupby('q_pn_unit_iid'):
            rid = str(rid_raw)
            pairs = []
            for val in ssub['q_pn_unit_contacting_pn_unit_iids']:
                pairs.extend(_parse_contacts_with_distance(val))
            rid_contacts_pairs[rid] = pairs

        dist_threshold = args.nucleic_acid_dist_threshold

        # Directed graph (u -> v): keep only when min_distance ≤ dist_threshold; both source and target must be nucleotides
        comp_dir_adj = {u: set() for u in atomic_nodes}
        for rid, comps in rid_to_comps.items():
            pairs = rid_contacts_pairs.get(rid, ())
            for u in comps:
                for target_iid, md in pairs:
                    if md is None or md > dist_threshold:
                        continue
                    for v in _split_components(target_iid):
                        if v in atomic_nodes and v != u:
                            comp_dir_adj[u].add(v)

        # Build undirected adjacency using OR semantics
        comp_adj = {u: set() for u in atomic_nodes}
        for u, vs in comp_dir_adj.items():
            for v in vs:
                comp_adj[u].add(v)
                comp_adj[v].add(u)

        # Compute connected components
        visited, components = set(), []
        for u in sorted(atomic_nodes, key=_natural_key):
            if u in visited:
                continue
            comp_set = set([u])
            dq = deque([u])
            visited.add(u)
            while dq:
                x = dq.popleft()
                for y in comp_adj.get(x, ()):
                    if y not in visited:
                        visited.add(y)
                        comp_set.add(y)
                        dq.append(y)
            components.append(comp_set)

        # Component label format: "(B_1, C_1, D_1)"
        label_to_nodes = {}
        comp_label_of_node = {}
        for comp_set in components:
            label = "(" + ", ".join(sorted(comp_set, key=_natural_key)) + ")"
            label_to_nodes[label] = comp_set
            for node in comp_set:
                comp_label_of_node[node] = label

        # Assign labels to each rid (including composite rids)
        rid_to_label = {}
        for rid, comps in rid_to_comps.items():
            nodes = set()
            for u in comps:
                label = comp_label_of_node.get(u)
                if label:
                    nodes |= label_to_nodes[label]
                else:
                    nodes.add(u)
            rid_to_label[rid] = "(" + ", ".join(sorted(nodes, key=_natural_key)) + ")"

        out = pd.Series(index=g.index, dtype=object)
        out.loc[na_mask] = g.loc[na_mask, 'q_pn_unit_iid'].astype(str).map(rid_to_label)
        return out
    
    def _sum_nucleic_acid_cluster_residues_per_pdb(g):
        mask = g['q_pn_unit_nucleic_acid_chain_cluster'].notna()
        if not mask.any():
            return pd.Series(0, index=g.index, dtype='int64')

        tmp = g.loc[mask, ['q_pn_unit_nucleic_acid_chain_cluster','q_pn_unit_iid','q_pn_unit_num_resolved_residues']].copy()
        tmp = tmp.drop_duplicates(subset=['q_pn_unit_nucleic_acid_chain_cluster','q_pn_unit_iid'])

        sums = tmp.groupby('q_pn_unit_nucleic_acid_chain_cluster')['q_pn_unit_num_resolved_residues'].sum()

        out = pd.Series(index=g.index, dtype='float')
        out.loc[mask] = g.loc[mask, 'q_pn_unit_nucleic_acid_chain_cluster'].map(sums)
        out = out.fillna(0)
        return out

    def _assign_ligand_clusters_per_pdb(g):
        # 비단백질 전체(소분자, 금속, 펩타이드, 핵산 포함)를 대상으로
        # 원자 체인 노드 + 핵산 클러스터 노드를 함께 사용하는 5Å 그래프를 구성한다.
        nonprot_mask = ~g['q_pn_unit_is_protein']
        if not nonprot_mask.any():
            return pd.Series(index=g.index, dtype=object)

        sub = g.loc[nonprot_mask, [
            'q_pn_unit_iid',
            'q_pn_unit_contacting_pn_unit_iids',
            'q_pn_unit_nucleic_acid_chain_cluster'
        ]].copy()

        # rid -> 원자 체인 컴포넌트
        rid_to_comps = {str(rid): tuple(_split_components(str(rid)))
                        for rid in sub['q_pn_unit_iid'].astype(str)}
        atomic_ids = set(c for comps in rid_to_comps.values() for c in comps)

        def _strip_parens(s: str) -> str:
            s = str(s).strip()
            if s.startswith("(") and s.endswith(")"):
                return s[1:-1].strip()
            return s

        # 정렬: 일반 원자 체인 먼저, 괄호형 그룹(핵산 클러스터)은 뒤로
        def _sort_key_for_items(s: str):
            s = str(s)
            is_group = s.startswith("(") and s.endswith(")")
            return (1 if is_group else 0, _natural_key(s))

        # 원자 체인 -> 소속 그룹 노드 매핑
        # - 핵산 체인은 해당 클러스터 라벨(예: "(C_1, D_1)")로 매핑
        # - 그 외는 자기 자신
        atomic_to_group = {a: a for a in atomic_ids}
        nuc_labels = sub.loc[sub['q_pn_unit_nucleic_acid_chain_cluster'].notna(),
                             'q_pn_unit_nucleic_acid_chain_cluster'].astype(str).unique()
        for cl in nuc_labels:
            for a in _split_components(_strip_parens(cl)):
                if a in atomic_ids:
                    atomic_to_group[a] = cl

        # 노드 집합: 모든 원자 체인을 그룹 노드로 사상한 결과
        nodes = set(atomic_to_group[a] for a in atomic_ids)

        # 체인별 (대상 chain_iid, min_distance) 목록 수집
        rid_contacts_pairs = {}
        for rid_raw, ssub in sub.groupby('q_pn_unit_iid'):
            rid = str(rid_raw)
            pairs = []
            for val in ssub['q_pn_unit_contacting_pn_unit_iids']:
                pairs.extend(_parse_contacts_with_distance(val))
            rid_contacts_pairs[rid] = pairs

        # 설정 임계값 Å 이하만 간선으로 유지(OR semantics), 노드는 그룹 기준
        dist_threshold = float(args.ligand_cluster_dist_threshold)
        adj = {n: set() for n in nodes}
        for rid, comps in rid_to_comps.items():
            pairs = rid_contacts_pairs.get(rid, ())
            for u in comps:
                if u not in atomic_ids:
                    continue
                nu = atomic_to_group[u]
                for target_iid, md in pairs:
                    if md is None or md > dist_threshold:
                        continue
                    for v in _split_components(target_iid):
                        if v in atomic_ids:
                            nv = atomic_to_group[v]
                            if nv != nu:
                                adj[nu].add(nv)
                                adj[nv].add(nu)

        # 그룹 노드 그래프의 연결 요소 계산
        visited, components = set(), []
        for n in sorted(nodes, key=_natural_key):
            if n in visited:
                continue
            comp = set([n])
            dq = deque([n])
            visited.add(n)
            while dq:
                x = dq.popleft()
                for y in adj.get(x, ()):
                    if y not in visited:
                        visited.add(y)
                        comp.add(y)
                        dq.append(y)
            components.append(comp)

        # 컴포넌트 라벨: 원자 체인은 그대로, 핵산 클러스터 노드는 "(C_1, D_1)" 그대로 하나의 아이템으로 유지
        label_to_nodes = {}
        comp_label_of_node = {}
        for comp in components:
            items = sorted(comp, key=_sort_key_for_items)
            label = "(" + ", ".join(items) + ")"
            label_to_nodes[label] = comp
            for n in comp:
                comp_label_of_node[n] = label

        # 각 rid에 동일한 클러스터 라벨 부여
        rid_to_label = {}
        for rid, comps in rid_to_comps.items():
            member_nodes = set(atomic_to_group.get(u, u) for u in comps)
            comp_nodes = set()
            for n in member_nodes:
                comp_nodes |= label_to_nodes[comp_label_of_node[n]]
            items = sorted(comp_nodes, key=_sort_key_for_items)
            rid_to_label[rid] = "(" + ", ".join(items) + ")"

        out = pd.Series(index=g.index, dtype=object)
        out.loc[nonprot_mask] = g.loc[nonprot_mask, 'q_pn_unit_iid'].astype(str).map(rid_to_label)
        return out
    
    def _count_atomic_members_in_ligand_cluster_label(label: str) -> int:
        # "(A_1, B_1, (C_1, D_1))" 형태의 문자열에서 원자 체인 개수를 센다.
        if label is None or (isinstance(label, float) and pd.isna(label)):
            return 0
        s = str(label).strip()
        if not s:
            return 0
        if not (s.startswith("(") and s.endswith(")")):
            return 1
        content = s[1:-1]
        items, buf, depth = [], [], 0
        for ch in content:
            if ch == '(':
                depth += 1
                buf.append(ch)
            elif ch == ')':
                depth -= 1
                buf.append(ch)
            elif ch == ',' and depth == 0:
                tok = ''.join(buf).strip()
                if tok:
                    items.append(tok)
                buf = []
            else:
                buf.append(ch)
        tok = ''.join(buf).strip()
        if tok:
            items.append(tok)
        count = 0
        for tok in items:
            tok = tok.strip()
            if not tok:
                continue
            if tok.startswith("(") and tok.endswith(")"):
                inner = tok[1:-1].strip()
                if not inner:
                    continue
                parts = [p.strip() for p in inner.split(',') if p.strip()]
                count += len(parts)
            else:
                count += 1
        return count
    
    def _compute_num_ligand_chain_in_cluster_per_pdb(g: pd.DataFrame) -> pd.Series:
        out = pd.Series(index=g.index, dtype='float')
        mask = (~g['q_pn_unit_is_protein']) & g['ligand_cluster'].notna()
        if not mask.any():
            return out
        label_to_count = {}
        for lab in g.loc[mask, 'ligand_cluster'].astype(str).unique():
            label_to_count[lab] = _count_atomic_members_in_ligand_cluster_label(lab)
        out.loc[mask] = g.loc[mask, 'ligand_cluster'].map(label_to_count).astype('float')
        return out
    
    def _compute_ligand_cluster_contacts_to_proteins_per_pdb(g: pd.DataFrame) -> pd.DataFrame:
        # 각 ligand_cluster에 대해 5Å 이내로 접촉하는 protein atomic chain들을 요약한다.
        prot_atomic = set()
        for iid in g.loc[g['q_pn_unit_is_protein'], 'q_pn_unit_iid'].astype(str):
            prot_atomic.update(_split_components(iid))

        out = pd.DataFrame(index=g.index, columns=[
            'ligand_cluster_contacting_protein_chains',
            'ligand_cluster_num_contacting_protein_chains'
        ])
        out['ligand_cluster_contacting_protein_chains'] = pd.NA
        out['ligand_cluster_num_contacting_protein_chains'] = pd.NA

        mask = (~g['q_pn_unit_is_protein']) & g['ligand_cluster'].notna()
        if not mask.any():
            return out

        # cluster -> set of contacting protein atomic chain ids within 5.0 Å
        cluster_to_prot = defaultdict(set)
        for cl, ssub in g.loc[mask, ['ligand_cluster','q_pn_unit_contacting_pn_unit_iids']].groupby('ligand_cluster'):
            acc = set()
            for val in ssub['q_pn_unit_contacting_pn_unit_iids']:
                for target_iid, md in _parse_contacts_with_distance(val):
                    if md is None or md > 5.0:
                        continue
                    for a in _split_components(target_iid):
                        if a in prot_atomic:
                            acc.add(a)
            cluster_to_prot[cl] = acc

        for cl, prot_set in cluster_to_prot.items():
            label = ("(" + _join_sorted(prot_set) + ")") if prot_set else ""
            cnt = int(len(prot_set))
            idx_mask = mask & (g['ligand_cluster'] == cl)
            out.loc[idx_mask, 'ligand_cluster_contacting_protein_chains'] = label
            out.loc[idx_mask, 'ligand_cluster_num_contacting_protein_chains'] = cnt

        return out
    
    # cluster nucleotides where they are near each other
    atomworks_parquet['q_pn_unit_nucleic_acid_chain_cluster'] = (
    atomworks_parquet.groupby('pdb_id', group_keys=False)
    .apply(_assign_nucleic_acid_chain_clusters_per_pdb, include_groups=False)
)
    atomworks_parquet['q_pn_unit_num_resolved_residues_in_nucleic_acid_chain_cluster'] = (
    atomworks_parquet.groupby('pdb_id', group_keys=False)
    .apply(_sum_nucleic_acid_cluster_residues_per_pdb, include_groups=False)
    .astype('int64')
)    
    atomworks_parquet["q_pn_unit_is_nuc_polymer"] = atomworks_parquet["q_pn_unit_nucleic_acid_chain_cluster"].notna() & (atomworks_parquet["q_pn_unit_num_resolved_residues_in_nucleic_acid_chain_cluster"] >= 2 * NUCLEIC_ACID_LIGANDS_MAX_RESIDUES)
    atomworks_parquet["q_pn_unit_is_nuc_ligand"] = atomworks_parquet["q_pn_unit_nucleic_acid_chain_cluster"].notna() & (atomworks_parquet["q_pn_unit_num_resolved_residues_in_nucleic_acid_chain_cluster"] < 2 * NUCLEIC_ACID_LIGANDS_MAX_RESIDUES)
   
    atomworks_parquet.to_parquet(f"{output_dir_path}/metadata_nuc_clustered.parquet")
    
    # 2) Delete small molecules that are covalently bonded to a protein
    # pdb_id별 protein chain iid 집합 구하기
    protein_iids_per_pdb = (
        atomworks_parquet[atomworks_parquet['q_pn_unit_is_protein']]
        .groupby('pdb_id')['q_pn_unit_iid']
        .apply(set)
        .to_dict()
    )

    # 문자열로 저장된 set을 파싱하는 함수
    def _parse_bonded_polymer_set(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return set()
        if isinstance(val, set):
            return val
        s = str(val).strip()
        if not s or s == 'set()':
            return set()
        try:
            parsed = ast.literal_eval(s)
            return set(parsed) if isinstance(parsed, (set, list, tuple)) else set()
        except Exception:
            return set()

    # 각 행의 bonded_polymer가 protein chain과 교집합이 있는지 확인
    def _has_bonded_protein(row):
        bonded = _parse_bonded_polymer_set(row['q_pn_unit_bonded_polymer_pn_units'])
        # 빈 set 체크
        if len(bonded) == 0:
            return False
        pdb_id = row['pdb_id']
        protein_iids = protein_iids_per_pdb.get(pdb_id, set())
        # bonded polymer 중 protein chain이 있으면 True
        return len(bonded & protein_iids) > 0

    # small molecule이면서 protein에 covalently bonded된 것 제외
    exclude_mask = (
        atomworks_parquet['q_pn_unit_is_small_molecule'] & 
        atomworks_parquet.apply(_has_bonded_protein, axis=1)
    )
    print(f"Excluding {exclude_mask.sum()} small molecules covalently bonded to proteins")
    atomworks_parquet = atomworks_parquet[~exclude_mask].reset_index(drop=True)
    
    # 3) Recompute ligand_cluster
    atomworks_parquet['ligand_cluster'] = (
    atomworks_parquet.groupby('pdb_id', group_keys=False)
    .apply(_assign_ligand_clusters_per_pdb, include_groups=False)
)
    
    # 3) For each ligand_cluster, compute contacting protein chains 
    tmp_lc = (
    atomworks_parquet.groupby('pdb_id', group_keys=False)
    .apply(_compute_ligand_cluster_contacts_to_proteins_per_pdb, include_groups=False)
    )
    mask_lig = (~atomworks_parquet['q_pn_unit_is_protein']) & atomworks_parquet['ligand_cluster'].notna()
    atomworks_parquet.loc[mask_lig, 'ligand_cluster_contacting_protein_chains'] = (
        tmp_lc.loc[mask_lig, 'ligand_cluster_contacting_protein_chains'].fillna("")
    )
    atomworks_parquet.loc[mask_lig, 'ligand_cluster_num_contacting_protein_chains'] = (
        tmp_lc.loc[mask_lig, 'ligand_cluster_num_contacting_protein_chains'].fillna(0).astype('int64')
    )
    # 4) Count ligand chains per ligand_cluster (expand nucleic clusters)
    tmp_num = (
    atomworks_parquet.groupby('pdb_id', group_keys=False)
    .apply(_compute_num_ligand_chain_in_cluster_per_pdb, include_groups=False)
    )
    atomworks_parquet.loc[mask_lig, 'num_ligand_chain_in_ligand_cluster'] = (
        tmp_num.loc[mask_lig].fillna(0).astype('int64')
    )
    
    atomworks_parquet.to_parquet(f"{output_dir_path}/metadata_nuc_ligand_clustered.parquet")
    
            
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet_path", default="/home/possu/jinho/datasets/atomworks_lmpnn_valset_filtered/metadata.parquet")
    ap.add_argument("--output_dir_path", default="/home/possu/jinho/datasets/atomworks_lmpnn_valset_filtered_re")
    ap.add_argument("--debug", default = False)
    ap.add_argument("--debug_num_pdb_ids", default = 1000)
    ap.add_argument("--nucleic_acid_dist_threshold", default = 4.0)
    ap.add_argument("--ligand_cluster_dist_threshold", default = 4.5)
    args = ap.parse_args()
    metadata_ligand_chain_clustering(args)

    
    


#     # 2) Recompute ligand-to-protein and protein-to-ligand summary columns
#     tmp = atomworks_parquet.groupby('pdb_id', group_keys=False).apply(_mark_cluster_contacts_to_proteins_per_pdb)
#     atomworks_parquet['contact_to_protein'] = tmp['contact_to_protein'].fillna(False)
#     atomworks_parquet['num_contacting_protein_chains'] = tmp['num_contacting_protein_chains'].fillna(0).astype('int64')
#     atomworks_parquet['contacting_protein_chains'] = tmp['contacting_protein_chains'].fillna("")

#     tmp2 = atomworks_parquet.groupby('pdb_id', group_keys=False).apply(_mark_protein_contacts_to_ligand_clusters_per_pdb)
#     atomworks_parquet['contact_to_ligand_cluster'] = tmp2['contact_to_ligand_cluster'].fillna(False)
#     atomworks_parquet['num_contacting_ligand_clusters'] = tmp2['num_contacting_ligand_clusters'].fillna(0).astype('int64')

#     # 3) Recompute second-shell cluster group labels
#     atomworks_parquet['second_shell_ligand_cluster'] = atomworks_parquet.groupby('pdb_id', group_keys=False).apply(_assign_second_shell_clusters_per_pdb)

#     # Save processed metadata parquet
#     atomworks_parquet.to_parquet(f"{output_dir_path}/metadata_ligand_clustered.parquet")
#     print(f"ligand chain clustering is done, saved at {output_dir_path}/metadata_ligand_clustered.parquet")

# def _assign_second_shell_clusters_per_pdb(g):
#     # Build cluster-level second-shell groups among ligand_clusters (including metals).
#     # Two clusters are grouped if any member of one is in the second shell of any member of the other (OR semantics).
#     mask = (~g['q_pn_unit_is_protein']) & g['ligand_cluster'].notna()
#     if not mask.any():
#         return pd.Series(index=g.index, dtype=object)

#     result = pd.Series(index=g.index, dtype=object)
#     df = g.loc[mask, ['ligand_cluster', 'q_pn_unit_iid', 'q_pn_unit_second_shell_pn_unit_iids']].copy()

#     # cluster -> set of member atomic chain ids (parsed from the label string)
#     cluster_to_members = {}
#     for cl in df['ligand_cluster'].astype(str).unique():
#         members = set(_split_components(cl))
#         cluster_to_members[cl] = members
#     atom_to_cluster = {}
#     for cl, members in cluster_to_members.items():
#         for a in members:
#             atom_to_cluster[a] = cl

#     # Chain-level second-shell map across all non-protein chains
#     rid_to_second_shell = {}
#     for rid_raw, ssub in g.loc[~g['q_pn_unit_is_protein'], ['q_pn_unit_iid','q_pn_unit_second_shell_pn_unit_iids']].groupby('q_pn_unit_iid'):
#         rid = str(rid_raw)
#         neigh = set()
#         for val in ssub['q_pn_unit_second_shell_pn_unit_iids']:
#             neigh.update(_parse_contacts(val))
#         rid_to_second_shell[rid] = neigh

#     # Cluster-level graph under OR semantics (edge if either side reports second-shell contact)
#     clusters = list(cluster_to_members.keys())
#     cl_adj = {cl: set() for cl in clusters}
#     for cl1, members in cluster_to_members.items():
#         for a in members:
#             for v in rid_to_second_shell.get(a, set()):
#                 cl2 = atom_to_cluster.get(v)
#                 if cl2 and cl2 != cl1:
#                     cl_adj[cl1].add(cl2)
#                     cl_adj[cl2].add(cl1)

#     # Connected components over clusters; each component is rendered as "(members), (members), ..."
#     visited = set()
#     comp_of_cluster = {}
#     comps = []
#     for cl in sorted(clusters, key=_natural_key):
#         if cl in visited:
#             continue
#         comp = set([cl])
#         dq = deque([cl])
#         visited.add(cl)
#         while dq:
#             x = dq.popleft()
#             for y in cl_adj.get(x, set()):
#                 if y not in visited:
#                     visited.add(y)
#                     comp.add(y)
#                     dq.append(y)
#         comps.append(comp)
#         for c in comp:
#             comp_of_cluster[c] = comp

#     # Build label string per connected component
#     def _cluster_group_label(comp):
#         group_labels = []
#         for c in sorted(comp, key=_natural_key):
#             members = cluster_to_members[c]
#             group_labels.append("(" + ", ".join(sorted(members, key=_natural_key)) + ")")
#         group_labels = sorted(group_labels, key=lambda s: re.split(r'(\d+)', s))
#         return ", ".join(group_labels)

#     # Map the component label back to all rows belonging to clusters in that component
#     for cl in clusters:
#         label_string = _cluster_group_label(comp_of_cluster[cl])
#         idx_mask = mask & (g['ligand_cluster'] == cl)
#         result.loc[idx_mask] = label_string

#     return result