import pandas as pd
import json
import ast
import re
from collections import deque

def split_components(iid):
    if not isinstance(iid, str):
        iid = str(iid)
    if ',' in iid:
        return [tok.strip() for tok in iid.split(',') if tok.strip()]
    return [iid.strip()] if iid else []

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]

def join_sorted(ids):
    ids = list(ids) if ids is not None else []
    if not ids:
        return ""
    return ", ".join(sorted(ids, key=natural_key))

def parse_contacts(val):
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

def parse_contacts_with_distance(val):
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

def compute_contacts_to_proteins_per_pdb(g: pd.DataFrame) -> pd.DataFrame:
    # Set of protein atomic chain ids
    prot_atomic = set()
    for iid in g.loc[g['q_pn_unit_is_protein'], 'q_pn_unit_iid'].astype(str):
        prot_atomic.update(split_components(iid))

    def _prot_contacts_from_val(val):
        # Parse contact list and intersect with protein atomic ids
        atomic = set()
        for cid in parse_contacts(val):
            atomic.update(split_components(cid))
        return atomic & prot_atomic

    out = pd.DataFrame(index=g.index, columns=['num_contacting_protein', 'contacting_protein_chains'])
    out['num_contacting_protein'] = 0
    out['contacting_protein_chains'] = ""

    sm_pep_metal_mask = (g['q_pn_unit_is_small_molecule'] | g['q_pn_unit_is_metal'] | g['q_pn_unit_is_peptide'])
    for idx, row in g.loc[sm_pep_metal_mask, ['q_pn_unit_contacting_pn_unit_iids']].iterrows():
        pcs = _prot_contacts_from_val(row['q_pn_unit_contacting_pn_unit_iids'])
        out.at[idx, 'num_contacting_protein'] = int(len(pcs))
        out.at[idx, 'contacting_protein_chains'] = ("(" + join_sorted(pcs) + ")") if pcs else ""

    out['num_contacting_protein'] = out['num_contacting_protein'].fillna(0).astype('int64')
    out['contacting_protein_chains'] = out['contacting_protein_chains'].fillna("")
    return out

def assign_nucleic_acid_chain_clusters_per_pdb(g: pd.DataFrame, nucleic_acid_dist_threshold: float):
    # Only DNA/RNA/RNA-DNA hybrid chains are considered
    na_mask = (g['q_pn_unit_is_DNA'] | g['q_pn_unit_is_RNA'] | g['q_pn_unit_is_RNA_DNA_hybrid'])
    if not na_mask.any():
        return pd.Series(index=g.index, dtype=object)

    sub = g.loc[na_mask, ['q_pn_unit_iid', 'q_pn_unit_contacting_pn_unit_iids']].copy()

    # Map chain id to atomic components (comma-separated)
    rid_to_comps = {str(rid): tuple(split_components(str(rid))) for rid in sub['q_pn_unit_iid'].astype(str)}
    atomic_nodes = set(c for comps in rid_to_comps.values() for c in comps)

    # Collect list of (target chain, min_distance) reported by each chain
    rid_contacts_pairs = {}
    for rid_raw, ssub in sub.groupby('q_pn_unit_iid'):
        rid = str(rid_raw)
        pairs = []
        for val in ssub['q_pn_unit_contacting_pn_unit_iids']:
            pairs.extend(parse_contacts_with_distance(val))
        rid_contacts_pairs[rid] = pairs

    dist_threshold = nucleic_acid_dist_threshold

    # Directed graph (u -> v): keep only when min_distance ≤ dist_threshold; both source and target must be nucleotides
    comp_dir_adj = {u: set() for u in atomic_nodes}
    for rid, comps in rid_to_comps.items():
        pairs = rid_contacts_pairs.get(rid, ())
        for u in comps:
            for target_iid, md in pairs:
                if md is None or md > dist_threshold:
                    continue
                for v in split_components(target_iid):
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
        for u in sorted(atomic_nodes, key=natural_key):
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
            label = "(" + ", ".join(sorted(comp_set, key=natural_key)) + ")"
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
            rid_to_label[rid] = "(" + ", ".join(sorted(nodes, key=natural_key)) + ")"

        out = pd.Series(index=g.index, dtype=object)
        out.loc[na_mask] = g.loc[na_mask, 'q_pn_unit_iid'].astype(str).map(rid_to_label)
        return out
    
def sum_nucleic_acid_cluster_residues_per_pdb(g):
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

def compute_nuc_cluster_contacts_to_proteins_per_pdb(g: pd.DataFrame) -> pd.DataFrame:
    # Set of protein atomic chain ids
    prot_atomic = set()
    for iid in g.loc[g['q_pn_unit_is_protein'], 'q_pn_unit_iid'].astype(str):
        prot_atomic.update(split_components(iid))

    out = pd.DataFrame(index=g.index, columns=['num_contacting_protein', 'contacting_protein_chains'])
    out['num_contacting_protein'] = pd.NA
    out['contacting_protein_chains'] = pd.NA

    mask = g['q_pn_unit_nucleic_acid_chain_cluster'].notna()
    if not mask.any():
        return out

    # For each cluster, union contacts of all chains in the cluster
    for cl, ssub in g.loc[mask].groupby('q_pn_unit_nucleic_acid_chain_cluster'):
        prot_contacts = set()
        for val in ssub['q_pn_unit_contacting_pn_unit_iids']:
            for cid in parse_contacts(val):
                prot_contacts.update(split_components(cid))
        prot_contacts &= prot_atomic

        cnt = int(len(prot_contacts))
        label = ("(" + join_sorted(prot_contacts) + ")") if prot_contacts else ""

        idx_mask = mask & (g['q_pn_unit_nucleic_acid_chain_cluster'] == cl)
        out.loc[idx_mask, 'num_contacting_protein'] = cnt
        out.loc[idx_mask, 'contacting_protein_chains'] = label

    return out