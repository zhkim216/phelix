import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import json
import os
from pathlib import Path

def get_heavy_atom_counts_from_smiles_cache(ccd_codes: list[str], smiles_cache: dict) -> dict[str, int]:
    """SMILES 캐시에서 heavy atom 수를 계산"""
    heavy_atom_counts = {}
    
    for ccd_code in tqdm(ccd_codes, desc="Counting heavy atoms"):
        smiles = smiles_cache.get(ccd_code)
        if smiles is None:
            continue
        
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                heavy_atom_counts[ccd_code] = mol.GetNumHeavyAtoms()
        except Exception as e:
            print(f"Failed for {ccd_code}: {e}")
            continue
    
    return heavy_atom_counts

def sum_heavy_atoms(res_names_str, heavy_atom_counts):
    if pd.isna(res_names_str) or not res_names_str:
        return None
    codes = res_names_str.split(',')
    total = 0
    for code in codes:
        count = heavy_atom_counts.get(code)
        if count is None:
            return None
        total += count
    return total

def has_protein_contact_too_close(contacting_json_str, pn_unit_is_protein_map, min_allowed_distance=2.4):
    """
    Check if any protein contact is too close (min_distance <= min_allowed_distance)
    Returns True if any protein contact is too close (should be excluded)
    """
    if pd.isna(contacting_json_str) or not contacting_json_str:
        return False
    
    try:
        contacts = json.loads(contacting_json_str)
        for contact in contacts:
            pn_unit_iid = contact.get('pn_unit_iid')
            min_distance = contact.get('min_distance')
            
            # Check if the pn_unit is a protein
            is_protein = pn_unit_is_protein_map.get(pn_unit_iid, False)
            
            if is_protein and min_distance is not None and min_distance <= min_allowed_distance:
                return True  # protein과 너무 가까움 -> 제외 대상
    except (json.JSONDecodeError, TypeError):
        return False
    
    return False

def main():
    METADATA_PATH = "/home/possu/jinho/datasets/atomworks_pdb_full_v3/metadata_seq_clustered_04.parquet"
    SMILES_CACHE_FILE = Path(__file__).parent / "ccd_smiles_cache_metadata_v3.json"
    HEAVY_ATOM_CACHE_FILE = Path(__file__).parent / "ccd_heavy_atom_counts.json"
    PASSED_CCD_CODES_PATH = Path(__file__).parent / "passed_ccd_codes_metadata_v3.txt"
    MIN_RESOLUTION_RATIO = 0.8
    MIN_ALLOWED_DISTANCE = 2.4
    DISTANCE_CUTOFF = 5.0
    
    print("Loading metadata...")
    metadata = pd.read_parquet(METADATA_PATH)

    # Load filter-passed ccd codes
    with open(PASSED_CCD_CODES_PATH, "r") as f:
        filter_passed_ccd_codes = [line.strip() for line in f.readlines()]
    
    # Get each pn_unit_iid contains any of the filtered ccd codes
    filter_passed_set = set(filter_passed_ccd_codes)
    
    metadata['q_pn_unit_has_filtered_ccd'] = metadata['q_pn_unit_non_polymer_res_names'].apply(
        lambda x: bool(set(x.split(',')) & filter_passed_set) if pd.notna(x) and x else False
    )   
               
    # Step 1: Load SMILES cache
    print("Loading SMILES cache...")
    with open(SMILES_CACHE_FILE, 'r') as f:
        smiles_cache = json.load(f)
    
    # Step 2: Calculate heavy atom counts (or load from cache)
    if os.path.exists(HEAVY_ATOM_CACHE_FILE):
        print("Loading heavy atom counts from cache...")
        with open(HEAVY_ATOM_CACHE_FILE, 'r') as f:
            ccd_heavy_atom_counts = json.load(f)
    else:
        ccd_codes_total_list = metadata['q_pn_unit_non_polymer_res_names'].unique().tolist()
        ccd_codes_total_list_dedup = []
        for ccd_code in ccd_codes_total_list:
            splited = ccd_code.split(",")
            for splited_ccd_code in splited:
                ccd_codes_total_list_dedup.append(splited_ccd_code)
                
        unique_ccd_codes_total = list(set(ccd_codes_total_list_dedup))
        print(f"unique_ccd_codes_total: {len(unique_ccd_codes_total)}")
        
        print("Computing heavy atom counts from SMILES...")
        ccd_heavy_atom_counts = get_heavy_atom_counts_from_smiles_cache(unique_ccd_codes_total, smiles_cache)
        with open(HEAVY_ATOM_CACHE_FILE, 'w') as f:
            json.dump(ccd_heavy_atom_counts, f)
        print(f"Saved to {HEAVY_ATOM_CACHE_FILE}")
    
    print(f"Heavy atom counts available for {len(ccd_heavy_atom_counts)} CCD codes")
            
    # Step 3: Map expected heavy atom counts to metadata
    metadata['q_pn_unit_expected_heavy_atoms_non_polymer'] = metadata['q_pn_unit_non_polymer_res_names'].apply(
    lambda x: sum_heavy_atoms(x, ccd_heavy_atom_counts)
)
    
    # Step 4: Apply basic filterings using resolution ratio # Todo: Add plinder's table A2 if there is a q_pn_unit_ligand_validity, if not, use min_resolution_ratio    
    metadata['q_pn_unit_resolution_ratio'] = (
        metadata['q_pn_unit_num_resolved_atoms'] / metadata['q_pn_unit_expected_heavy_atoms_non_polymer']
    )
    
    # Step 5: Show only biologically meaningful small molecules if resolution ratio is 80% or higher
    metadata['q_pn_unit_is_biologically_meaningful_small_molecule'] = (
        (~metadata['q_pn_unit_is_polymer']) & 
        (metadata['q_pn_unit_has_filtered_ccd']) &
        (metadata['q_pn_unit_resolution_ratio'] >= MIN_RESOLUTION_RATIO)
    )
    
    metadata_bm_sm = metadata[metadata['q_pn_unit_is_biologically_meaningful_small_molecule']]
    
    # Check how many small molecules are filtered out
    before_count = len(metadata[
        (~metadata['q_pn_unit_is_polymer']) & 
        (metadata['q_pn_unit_has_filtered_ccd'])
    ])
    print(f"\nBefore {MIN_RESOLUTION_RATIO*100}% resolution filtering: {before_count}")
    print(f"After {MIN_RESOLUTION_RATIO*100}% resolution filtering: {len(metadata_bm_sm)}")
    print(f"Filtered out: {before_count - len(metadata_bm_sm)}")
                    
    
    # Step 6: Filter biologically meaningful small molecules by protein contact
    print(f"\nStep 6: Filtering biologically meaningful small molecules by protein contact...")
    print(f"  - Must have protein contact within {DISTANCE_CUTOFF} Å")
    print(f"  - If min_distance <= {MIN_ALLOWED_DISTANCE} Å, mark as maybe covalently linked")
    
    # Create a mapping of (pdb_id, assembly_id, pn_unit_iid) -> is_protein
    metadata['_key'] = metadata['pdb_id'] + '_' + metadata['assembly_id'].astype(str) + '_' + metadata['q_pn_unit_iid']
    pn_unit_is_protein_map = metadata.set_index('_key')['q_pn_unit_is_protein'].to_dict()
    
    # Step 6-1: Parse all JSON strings at once (vectorized)
    print("Parsing contact JSON strings...")
    def parse_contacts_json(json_str):
        if pd.isna(json_str) or not json_str:
            return []
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return []
    
    metadata['_parsed_contacts'] = metadata['q_pn_unit_contacting_pn_unit_iids'].apply(parse_contacts_json)
    
    # Step 6-2: Check if there's a valid protein contact (within cutoff) and if it's maybe covalently linked
    print("Checking protein contact distances...")
    
    def check_protein_contact(row):
        """
        Returns tuple of (has_valid_contact, is_maybe_covalently_linked):
        - has_valid_contact: True if there is at least one protein contact within DISTANCE_CUTOFF
        - is_maybe_covalently_linked: True if any protein contact has min_distance <= 2.4 Å
        """
        contacts = row['_parsed_contacts']
        if not contacts:
            return (False, False)
        
        pdb_id = row['pdb_id']
        assembly_id = row['assembly_id']
        
        has_valid_contact = False
        is_maybe_covalently_linked = False
        
        for contact in contacts:
            pn_unit_iid = contact.get('pn_unit_iid')
            min_distance = contact.get('min_distance')
            
            key = f"{pdb_id}_{assembly_id}_{pn_unit_iid}"
            is_protein = pn_unit_is_protein_map.get(key, False)
            
            if is_protein and min_distance is not None:
                # protein과 접촉하고 있고, min_distance <= DISTANCE_CUTOFF
                if min_distance <= DISTANCE_CUTOFF:
                    has_valid_contact = True
                    # min_distance <= 2.4 Å 이면 covalent로 추정
                    if min_distance <= MIN_ALLOWED_DISTANCE:
                        is_maybe_covalently_linked = True
        
        return (has_valid_contact, is_maybe_covalently_linked)
    
    # Apply only to biologically meaningful small molecules
    bm_sm_mask = metadata['q_pn_unit_is_biologically_meaningful_small_molecule']
    print(f"Checking {bm_sm_mask.sum()} biologically meaningful small molecule entries...")
    
    metadata['_has_valid_protein_contact'] = False
    metadata['q_pn_unit_is_maybe_covalently_linked_to_protein'] = False
    
    # Apply and unpack tuple results
    contact_results = metadata.loc[bm_sm_mask].apply(check_protein_contact, axis=1)
    metadata.loc[bm_sm_mask, '_has_valid_protein_contact'] = contact_results.apply(lambda x: x[0])
    metadata.loc[bm_sm_mask, 'q_pn_unit_is_maybe_covalently_linked_to_protein'] = contact_results.apply(lambda x: x[1])
    
    # Update small molecules: must have valid protein contact
    before_count = bm_sm_mask.sum()
    metadata['q_pn_unit_is_biologically_meaningful_small_molecule'] = (
        metadata['q_pn_unit_is_biologically_meaningful_small_molecule'] & 
        metadata['_has_valid_protein_contact']
    )
    after_count = metadata['q_pn_unit_is_biologically_meaningful_small_molecule'].sum()
    
    print(f"\nBefore protein contact filtering: {before_count}")
    print(f"After protein contact filtering: {after_count}")
    print(f"Filtered out (no valid protein contact): {before_count - after_count}")
    
    # Drop temporary columns
    metadata.drop(columns=['_key', '_parsed_contacts', '_has_valid_protein_contact'], inplace=True)
    
    # Show final statistics
    print(f"\n=== Final Statistics ===")
    print(f"Number of biologically meaningful small molecules: {metadata['q_pn_unit_is_biologically_meaningful_small_molecule'].sum()}")
    print(f"Number of maybe covalently linked to protein: {metadata['q_pn_unit_is_maybe_covalently_linked_to_protein'].sum()}")

    # Save filtered metadata
    output_path = "/home/possu/jinho/datasets/atomworks_pdb_full_v3/metadata_seq_clustered_04_filtered_250205.parquet"
    print(f"\nSaving filtered metadata to {output_path}...")
    metadata.to_parquet(output_path)
    print("Done!")

if __name__ == '__main__':
    main()