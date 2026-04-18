
import pandas as pd
import requests
from rdkit import Chem
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

from atomworks.io.tools.rdkit import ccd_code_to_rdkit

from pathlib import Path

SMILES_CACHE_FILE = Path(__file__).parent / "ccd_smiles_cache_metadata_v3.json"

def fetch_smiles_from_rcsb(ccd_code):
    """Fetch SMILES from RCSB PDB API for a CCD code."""
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ccd_code}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            desc = data.get('rcsb_chem_comp_descriptor', {})
            # Prefer smilesstereo (with stereochemistry) over smiles
            smiles = desc.get('smilesstereo') or desc.get('smiles')
            return smiles
        else:
            return None
    except Exception as e:
        return None


def generate_smiles_from_ccd(ccd_code):
    """Generate SMILES from CCD code using atomworks (fallback for when RCSB has no SMILES)."""
    try:
        mol = ccd_code_to_rdkit(ccd_code, fix_stereochemistry=False, return_atom_array=False)
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            return smiles
        return None
    except Exception as e:
        return None


def load_smiles_cache():
    """Load SMILES cache from file."""
    if os.path.exists(SMILES_CACHE_FILE):
        with open(SMILES_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_smiles_cache(cache):
    """Save SMILES cache to file."""
    with open(SMILES_CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def fetch_smiles_for_ccd(ccd_code, cache):
    """Fetch SMILES for a CCD code, using cache if available."""
    if ccd_code in cache:
        return ccd_code, cache[ccd_code], "cache"
    
    smiles = fetch_smiles_from_rcsb(ccd_code)
    if smiles:
        return ccd_code, smiles, "api"
    else:
        return ccd_code, None, "failed"


def fetch_all_smiles(ccd_codes, num_workers=32):
    """Fetch SMILES for all CCD codes using parallel HTTP requests."""
    cache = load_smiles_cache()
    
    # Find codes not in cache
    uncached_codes = [c for c in ccd_codes if c not in cache]
    print(f"  - Cached: {len(ccd_codes) - len(uncached_codes)}")
    print(f"  - To fetch: {len(uncached_codes)}")
    
    if uncached_codes:
        # Fetch uncached codes in parallel
        def fetch_single(ccd_code):
            smiles = fetch_smiles_from_rcsb(ccd_code)
            return ccd_code, smiles
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(fetch_single, uncached_codes),
                total=len(uncached_codes),
                desc="Fetching SMILES from RCSB"
            ))
        
        # Update cache with new results
        for ccd_code, smiles in results:
            cache[ccd_code] = smiles  # None if failed
        
        # Save updated cache
        save_smiles_cache(cache)
        print(f"  - Cache updated and saved")
    
    return cache

def main():
    metadata = pd.read_parquet("/home/possu/jinho/datasets/atomworks_pdb_full_v3/metadata_seq_clustered_04.parquet")

    # Take only non-polymer ccd codes
    non_polymer_metadata = metadata[~metadata['q_pn_unit_is_polymer']]
    
    ccd_codes = non_polymer_metadata['q_pn_unit_non_polymer_res_names'].unique().tolist()

    ccd_codes_list = []
    for ccd_code in ccd_codes:
        splited = ccd_code.split(",")
        for splited_ccd_code in splited:
            ccd_codes_list.append(splited_ccd_code)
            
    unique_ccd_codes = list(set(ccd_codes_list))
    print(f"unique_ccd_codes: {len(unique_ccd_codes)}")
    
    # Step 1: Fetch SMILES from RCSB API (thread-safe, no RDKit involved)
    print(f"\n=== Step 1: Fetching SMILES from RCSB API ===")
    smiles_cache = fetch_all_smiles(unique_ccd_codes, num_workers=32)
    
    # Count stats
    has_smiles = sum(1 for c in unique_ccd_codes if smiles_cache.get(c) is not None)
    no_smiles_codes = [c for c in unique_ccd_codes if smiles_cache.get(c) is None]
    print(f"  - SMILES available: {has_smiles}/{len(unique_ccd_codes)}")
    print(f"  - No SMILES: {len(no_smiles_codes)}")
    
    # Step 1.5: Try atomworks fallback for codes without SMILES
    if no_smiles_codes:
        print(f"\n=== Step 1.5: Generating SMILES via atomworks for {len(no_smiles_codes)} codes ===")
        generated_count = 0
        for ccd_code in tqdm(no_smiles_codes, desc="Generating SMILES via atomworks"):
            smiles = generate_smiles_from_ccd(ccd_code)
            if smiles:
                smiles_cache[ccd_code] = smiles
                generated_count += 1
        
        # Save updated cache
        save_smiles_cache(smiles_cache)
        print(f"  - Generated: {generated_count}/{len(no_smiles_codes)}")
        
        # Recount
        has_smiles = sum(1 for c in unique_ccd_codes if smiles_cache.get(c) is not None)
        print(f"  - SMILES available (after fallback): {has_smiles}/{len(unique_ccd_codes)}")
    
    print(f"\n=== DONE ===")
    print(f"Total unique CCD codes: {len(unique_ccd_codes)}")
    print(f"SMILES cached: {has_smiles}")
    print(f"Cache file: {SMILES_CACHE_FILE}")


if __name__ == '__main__':
    main()
