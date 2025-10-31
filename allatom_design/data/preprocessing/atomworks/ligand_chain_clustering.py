import pandas as pd
from atomworks.enums import ChainTypeInfo, ChainType
from atomworks.constants import METAL_ELEMENTS
import json, ast, re
from collections import defaultdict, deque
import argparse

def metadata_ligand_chain_clustering(input_parquet_path: str=None,
                                    output_dir_path: str=None):

    # Load the original metadata parquet
    atomworks_parquet = pd.read_parquet(input_parquet_path)

    ### Proteins
    protein_chain_types = ChainTypeInfo.PROTEINS
    protein_chain_type_values = [chain_type.value for chain_type in protein_chain_types]

    ### Nucleic acids
    DNA_chain_type_values = [ChainType.DNA.value]
    RNA_chain_type_values = [ChainType.RNA.value]
    RNA_DNA_hybrid_chain_type_values = [ChainType.DNA_RNA_HYBRID.value]

    ### Ligands
    ligand_chain_types = ChainTypeInfo.NON_POLYMERS
    ligand_chain_type_values = [chain_type.value for chain_type in ligand_chain_types]

    # add "is_protein" & "is_peptide" column, following the definition in Atomworks
    atomworks_parquet["q_pn_unit_is_protein"] = (atomworks_parquet["q_pn_unit_type"].isin(protein_chain_type_values)) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] >= 20)
    atomworks_parquet["q_pn_unit_is_peptide"] = (atomworks_parquet["q_pn_unit_type"].isin(protein_chain_type_values)) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] < 20)

    # DNA polymer & ligand columns, following the definition in Plinder
    atomworks_parquet["q_pn_unit_is_DNA_polymer"] = atomworks_parquet["q_pn_unit_type"].isin(DNA_chain_type_values) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] > 10)
    atomworks_parquet["q_pn_unit_is_DNA_ligand"] = atomworks_parquet["q_pn_unit_type"].isin(DNA_chain_type_values) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] <= 10)

    # RNA polymer & ligand columns, following the definition in Plinder
    atomworks_parquet["q_pn_unit_is_RNA_polymer"] = atomworks_parquet["q_pn_unit_type"].isin(RNA_chain_type_values) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] > 10)
    atomworks_parquet["q_pn_unit_is_RNA_ligand"] = atomworks_parquet["q_pn_unit_type"].isin(RNA_chain_type_values) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] <= 10)

    # RNA-DNA hybrid polymer & ligand columns, following the definition in Plinder
    atomworks_parquet["q_pn_unit_is_RNA_DNA_hybrid_polymer"] = atomworks_parquet["q_pn_unit_type"].isin(RNA_DNA_hybrid_chain_type_values) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] > 10)
    atomworks_parquet["q_pn_unit_is_RNA_DNA_hybrid_ligand"] = atomworks_parquet["q_pn_unit_type"].isin(RNA_DNA_hybrid_chain_type_values) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] <= 10)

    # Small molecule ligands & small molecule - metal complexes
    atomworks_parquet["q_pn_unit_is_small_molecule"] = (atomworks_parquet["q_pn_unit_type"].isin(ligand_chain_type_values)) & (atomworks_parquet["q_pn_unit_is_metal"] == False)

    # Take only pdb_id with at least one protein chain
    mask = atomworks_parquet.groupby('pdb_id')['q_pn_unit_is_protein'].transform('any')
    atomworks_parquet = atomworks_parquet.loc[mask].reset_index(drop=True)

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

    def _assign_ligand_clusters_per_pdb(g):
        # Non-protein (including metals): build an undirected ligand graph where an edge exists
        # if the minimum heavy-atom distance between two ligand chains is <= 5.0 Å.
        # Use OR semantics (edge kept even if reported only by one side). Clusters are connected components.
        nonprot_mask = ~g['q_pn_unit_is_protein']
        if not nonprot_mask.any():
            return pd.Series(index=g.index, dtype=object)

        sub = g.loc[nonprot_mask, ['q_pn_unit_iid', 'q_pn_unit_contacting_pn_unit_iids']].copy()
        # rid -> atomic components
        rid_to_comps = {str(rid): tuple(_split_components(str(rid))) for rid in sub['q_pn_unit_iid'].astype(str)}
        atomic_nodes = set(c for comps in rid_to_comps.values() for c in comps)

        # Collect (contact_iid, min_distance) pairs per rid
        rid_contacts_pairs = {}
        for rid_raw, ssub in sub.groupby('q_pn_unit_iid'):
            rid = str(rid_raw)
            pairs = []
            for val in ssub['q_pn_unit_contacting_pn_unit_iids']:
                pairs.extend(_parse_contacts_with_distance(val))
            rid_contacts_pairs[rid] = pairs

        dist_threshold = 5.0

        # Directed edges (u -> v): u is an atomic component of rid; v is a non-protein atomic id in the same pdb
        # Keep only contacts with min_distance <= 5.0 Å
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

        # Undirected adjacency under OR semantics
        comp_adj = {u: set() for u in atomic_nodes}
        for u, vs in comp_dir_adj.items():
            for v in vs:
                comp_adj[u].add(v)
                comp_adj[v].add(u)

        # Connected components over atomic nodes
        visited, components = set(), []
        for u in sorted(atomic_nodes, key=_natural_key):
            if u in visited:
                continue
            comp_set = set([u])
            dq = deque([u])
            visited.add(u)
            while dq:
                x = dq.popleft()
                for y in comp_adj.get(x, ( )):
                    if y not in visited:
                        visited.add(y)
                        comp_set.add(y)
                        dq.append(y)
            components.append(comp_set)

        # Component labels
        label_to_nodes = {}
        comp_label_of_node = {}
        for comp_set in components:
            label = ", ".join(sorted(comp_set, key=_natural_key))
            label_to_nodes[label] = comp_set
            for node in comp_set:
                comp_label_of_node[node] = label

        # Assign labels per rid by union of its atomic components' component nodes
        rid_to_label = {}
        for rid, comps in rid_to_comps.items():
            nodes = set()
            for u in comps:
                label = comp_label_of_node.get(u)
                if label:
                    nodes |= label_to_nodes[label]
                else:
                    nodes.add(u)
            rid_to_label[rid] = ", ".join(sorted(nodes, key=_natural_key))

        out = pd.Series(index=g.index, dtype=object)
        out.loc[nonprot_mask] = g.loc[nonprot_mask, 'q_pn_unit_iid'].astype(str).map(rid_to_label)
        return out

    def _mark_cluster_contacts_to_proteins_per_pdb(g):
        prot_mask = g['q_pn_unit_is_protein']

        prot_atomic = set()
        for iid in g.loc[prot_mask, 'q_pn_unit_iid'].astype(str):
            prot_atomic.update(_split_components(iid))

        cluster_to_prot = defaultdict(set)
        sub = g.loc[(~g['q_pn_unit_is_protein']) & g['ligand_cluster'].notna(),
                    ['ligand_cluster', 'q_pn_unit_contacting_pn_unit_iids']]
        for _, row in sub.iterrows():
            contacts = set(_parse_contacts(row['q_pn_unit_contacting_pn_unit_iids']))
            cluster_to_prot[row['ligand_cluster']].update(contacts & prot_atomic)

        out = pd.DataFrame(index=g.index, columns=[
            'contact_to_protein',
            'num_contacting_protein_chains',
            'contacting_protein_chains'
        ])
        mask = (~g['q_pn_unit_is_protein']) & g['ligand_cluster'].notna()
        out.loc[mask, 'contact_to_protein'] = g.loc[mask, 'ligand_cluster'] \
            .map(lambda c: len(cluster_to_prot.get(c, set())) > 0)
        out.loc[mask, 'num_contacting_protein_chains'] = g.loc[mask, 'ligand_cluster'] \
            .map(lambda c: len(cluster_to_prot.get(c, set())))
        out.loc[mask, 'contacting_protein_chains'] = g.loc[mask, 'ligand_cluster'] \
            .map(lambda c: _join_sorted(cluster_to_prot.get(c, set())))
        return out

    def _mark_protein_contacts_to_ligand_clusters_per_pdb(g):
        prot_mask = g['q_pn_unit_is_protein']

        prot_atomic = set()
        for iid in g.loc[prot_mask, 'q_pn_unit_iid'].astype(str):
            prot_atomic.update(_split_components(iid))

        ligand_mask = (~g['q_pn_unit_is_protein']) & g['ligand_cluster'].notna()
        cluster_to_prot_atomic = defaultdict(set)
        for _, row in g.loc[ligand_mask, ['ligand_cluster','q_pn_unit_contacting_pn_unit_iids']].iterrows():
            contacts = set(_parse_contacts(row['q_pn_unit_contacting_pn_unit_iids']))
            cluster_to_prot_atomic[row['ligand_cluster']].update(contacts & prot_atomic)

        prot_atomic_to_clusters = defaultdict(set)
        for cl, ps in cluster_to_prot_atomic.items():
            for p in ps:
                prot_atomic_to_clusters[p].add(cl)

        out = pd.DataFrame(index=g.index, columns=['contact_to_ligand_cluster','num_contacting_ligand_clusters'])
        for idx, rid in g.loc[prot_mask, 'q_pn_unit_iid'].astype(str).items():
            comps = _split_components(rid)
            touching = set()
            for comp in comps:
                touching |= prot_atomic_to_clusters.get(comp, set())
            out.at[idx, 'contact_to_ligand_cluster'] = len(touching) > 0
            out.at[idx, 'num_contacting_ligand_clusters'] = len(touching)
        return out

    def _assign_second_shell_clusters_per_pdb(g):
        # Build cluster-level second-shell groups among ligand_clusters (including metals).
        # Two clusters are grouped if any member of one is in the second shell of any member of the other (OR semantics).
        mask = (~g['q_pn_unit_is_protein']) & g['ligand_cluster'].notna()
        if not mask.any():
            return pd.Series(index=g.index, dtype=object)

        result = pd.Series(index=g.index, dtype=object)
        df = g.loc[mask, ['ligand_cluster', 'q_pn_unit_iid', 'q_pn_unit_second_shell_pn_unit_iids']].copy()

        # cluster -> set of member atomic chain ids (parsed from the label string)
        cluster_to_members = {}
        for cl in df['ligand_cluster'].astype(str).unique():
            members = set(_split_components(cl))
            cluster_to_members[cl] = members
        atom_to_cluster = {}
        for cl, members in cluster_to_members.items():
            for a in members:
                atom_to_cluster[a] = cl

        # Chain-level second-shell map across all non-protein chains
        rid_to_second_shell = {}
        for rid_raw, ssub in g.loc[~g['q_pn_unit_is_protein'], ['q_pn_unit_iid','q_pn_unit_second_shell_pn_unit_iids']].groupby('q_pn_unit_iid'):
            rid = str(rid_raw)
            neigh = set()
            for val in ssub['q_pn_unit_second_shell_pn_unit_iids']:
                neigh.update(_parse_contacts(val))
            rid_to_second_shell[rid] = neigh

        # Cluster-level graph under OR semantics (edge if either side reports second-shell contact)
        clusters = list(cluster_to_members.keys())
        cl_adj = {cl: set() for cl in clusters}
        for cl1, members in cluster_to_members.items():
            for a in members:
                for v in rid_to_second_shell.get(a, set()):
                    cl2 = atom_to_cluster.get(v)
                    if cl2 and cl2 != cl1:
                        cl_adj[cl1].add(cl2)
                        cl_adj[cl2].add(cl1)

        # Connected components over clusters; each component is rendered as "(members), (members), ..."
        visited = set()
        comp_of_cluster = {}
        comps = []
        for cl in sorted(clusters, key=_natural_key):
            if cl in visited:
                continue
            comp = set([cl])
            dq = deque([cl])
            visited.add(cl)
            while dq:
                x = dq.popleft()
                for y in cl_adj.get(x, set()):
                    if y not in visited:
                        visited.add(y)
                        comp.add(y)
                        dq.append(y)
            comps.append(comp)
            for c in comp:
                comp_of_cluster[c] = comp

        # Build label string per connected component
        def _cluster_group_label(comp):
            group_labels = []
            for c in sorted(comp, key=_natural_key):
                members = cluster_to_members[c]
                group_labels.append("(" + ", ".join(sorted(members, key=_natural_key)) + ")")
            group_labels = sorted(group_labels, key=lambda s: re.split(r'(\d+)', s))
            return ", ".join(group_labels)

        # Map the component label back to all rows belonging to clusters in that component
        for cl in clusters:
            label_string = _cluster_group_label(comp_of_cluster[cl])
            idx_mask = mask & (g['ligand_cluster'] == cl)
            result.loc[idx_mask] = label_string

        return result

    # 1) Recompute ligand_cluster
    atomworks_parquet['ligand_cluster'] = atomworks_parquet.groupby('pdb_id', group_keys=False).apply(_assign_ligand_clusters_per_pdb)

    # 2) Recompute ligand-to-protein and protein-to-ligand summary columns
    tmp = atomworks_parquet.groupby('pdb_id', group_keys=False).apply(_mark_cluster_contacts_to_proteins_per_pdb)
    atomworks_parquet['contact_to_protein'] = tmp['contact_to_protein'].fillna(False)
    atomworks_parquet['num_contacting_protein_chains'] = tmp['num_contacting_protein_chains'].fillna(0).astype('int64')
    atomworks_parquet['contacting_protein_chains'] = tmp['contacting_protein_chains'].fillna("")

    tmp2 = atomworks_parquet.groupby('pdb_id', group_keys=False).apply(_mark_protein_contacts_to_ligand_clusters_per_pdb)
    atomworks_parquet['contact_to_ligand_cluster'] = tmp2['contact_to_ligand_cluster'].fillna(False)
    atomworks_parquet['num_contacting_ligand_clusters'] = tmp2['num_contacting_ligand_clusters'].fillna(0).astype('int64')

    # 3) Recompute second-shell cluster group labels
    atomworks_parquet['second_shell_ligand_cluster'] = atomworks_parquet.groupby('pdb_id', group_keys=False).apply(_assign_second_shell_clusters_per_pdb)

    # Save processed metadata parquet
    atomworks_parquet.to_parquet(f"{output_dir_path}/metadata_ligand_clustered.parquet")
    print(f"ligand chain clustering is done, saved at {output_dir_path}/metadata_ligand_clustered.parquet")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet_path", default="/scratch/users/zhkim216/datasets/atomworks_lmpnn/metadata.parquet")
    ap.add_argument("--output_dir_path", default="/scratch/users/zhkim216/datasets/atomworks_lmpnn")
    args = ap.parse_args()
    metadata_ligand_chain_clustering(input_parquet_path = args.input_parquet_path, output_dir_path = args.output_dir_path)

    