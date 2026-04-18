import pandas as pd
import requests
from rdkit import Chem
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

from atomworks.io.tools.rdkit import ccd_code_to_rdkit

# Cache file for SMILES (to avoid repeated API calls)
SMILES_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ccd_smiles_cache_metadata_v3.json")


### Table A3 Non-artifact ligand classification criteria (from plinder)
def get_longest_linear_hydrocarbon_linker(mol, max_count=50):
    """Get length of longest unbranched hydrocarbon linker"""
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SYMMRINGS)
        link_unit = "[#6D2R0]"
        for i in range(max_count):
            chain_smarts = "~".join([link_unit] * (i + 1))
            if len(mol.GetSubstructMatches(Chem.MolFromSmarts(chain_smarts))) == 0:
                return i
        return -1
    except:
        return -1


def is_non_artifact_ligand(smiles):
    """
    Check if ligand passes Table A3 criteria (non-artifact).
    Returns True if ligand is valid (non-artifact), False otherwise.
    Excluded absolute charge criteria because there are biologically important ligands with absolute charge > 2 (ADP, ATP, etc.).    
    Criteria:
        - IS A SINGLE ATOM (ION) = FALSE
        - NON-H ATOM COUNT > 5
        - C ATOM COUNT > 2
        - UNBRANCHED HYDROCARBON LINKER LENGTH <= 12
        - UNSPECIFIED ATOM COUNT = 0
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return False
    
    numHA = mol.GetNumHeavyAtoms()
    # 1. Single atom check (is_single_atom_or_ion)
    if numHA == 1:
        return False
    
    # 2. Non-H atom count > 5
    if numHA <= 5:
        return False
    
    # 3. C atom count > 2
    carbon = Chem.MolFromSmarts("[#6]")
    numC = len(mol.GetSubstructMatches(carbon))
    if numC <= 2:
        return False
    
    # 4. Unbranched hydrocarbon linker <= 12
    linker_len = get_longest_linear_hydrocarbon_linker(mol)
    if linker_len > 12:
        return False
    
    # 5. Unspecified atom count = 0 (atomic number 0 = wildcard/query atom)
    unspecified_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
    if unspecified_count > 0:
        return False
    
    return True


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


def filter_by_table_a3(ccd_codes, smiles_cache):
    """Filter CCD codes by Table A3 criteria using cached SMILES."""
    passed = []
    failed = []
    
    for ccd_code in tqdm(ccd_codes, desc="Filtering by Table A3"):
        smiles = smiles_cache.get(ccd_code)
        
        if smiles is None:
            failed.append((ccd_code, "no_smiles"))
            continue
        
        try:
            if is_non_artifact_ligand(smiles):
                passed.append(ccd_code)
            else:
                failed.append((ccd_code, "table_a3_failed"))
        except Exception as e:
            failed.append((ccd_code, str(e)))
    
    return passed, failed


def main():
    print("Loading metadata...")
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

    with open("/home/possu/jinho/datasets/ccd_reference_lists/plinder_artifact_ccd_codes.txt", "r") as f:
        artifact_ccd_codes = [line.strip() for line in f.readlines()]    
    
    with open("/home/possu/jinho/datasets/ccd_reference_lists/plinder_artifact_ccd_codes.txt", "w") as f:
        for ccd_code in artifact_ccd_codes:
            f.write(f"{ccd_code}\n")

    unique_ccd_codes_filtered1 = [ccd_code for ccd_code in unique_ccd_codes if ccd_code not in artifact_ccd_codes]
    print(f"unique_ccd_codes_filtered1: {len(unique_ccd_codes_filtered1)}")

    # Step 1: Fetch SMILES from RCSB API (thread-safe, no RDKit involved)
    print(f"\n=== Step 1: Fetching SMILES from RCSB API ===")
    smiles_cache = fetch_all_smiles(unique_ccd_codes_filtered1, num_workers=32)
    
    # Count stats
    has_smiles = sum(1 for c in unique_ccd_codes_filtered1 if smiles_cache.get(c) is not None)
    no_smiles_codes = [c for c in unique_ccd_codes_filtered1 if smiles_cache.get(c) is None]
    print(f"  - SMILES available: {has_smiles}/{len(unique_ccd_codes_filtered1)}")
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
        has_smiles = sum(1 for c in unique_ccd_codes_filtered1 if smiles_cache.get(c) is not None)
        print(f"  - SMILES available (after fallback): {has_smiles}/{len(unique_ccd_codes_filtered1)}")

    # Step 2: Filter by Table A3 criteria (using SMILES only, no RDKit mol loading)
    print(f"\n=== Step 2: Filtering by Table A3 criteria ===")
    passed, failed = filter_by_table_a3(unique_ccd_codes_filtered1, smiles_cache)

    unique_ccd_codes_filtered2 = passed

    print(f"\n=== FINAL RESULTS ===")
    print(f"unique_ccd_codes_filtered2: {len(unique_ccd_codes_filtered2)} passed, {len(failed)} failed")
    print(f"  - Failed (no_smiles): {len([f for f in failed if f[1] == 'no_smiles'])}")
    print(f"  - Failed (table_a3): {len([f for f in failed if f[1] == 'table_a3_failed'])}")
    print(f"  - Failed (other): {len([f for f in failed if f[1] not in ['no_smiles', 'table_a3_failed']])}")
    
    # Save results
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_ccd_codes_filtered2.txt")
    with open(output_file, "w") as f:
        for ccd in sorted(unique_ccd_codes_filtered2):
            f.write(f"{ccd}\n")
    print(f"Saved passed CCD codes to {output_file}")
    
    # Save passed CCD codes list
    passed_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "passed_ccd_codes_metadata_v3.txt")
    with open(passed_file, "w") as f:
        for ccd in sorted(unique_ccd_codes_filtered2):
            f.write(f"{ccd}\n")
    print(f"Saved passed CCD codes to {passed_file}")
    
    # Save failed CCD codes list
    failed_txt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "failed_ccd_codes_metadata_v3.txt")
    with open(failed_txt_file, "w") as f:
        for ccd_code, reason in sorted(failed, key=lambda x: x[0]):
            f.write(f"{ccd_code}\n")
    print(f"Saved failed CCD codes to {failed_txt_file}")
    
    # Save failed codes with reasons (JSON format)
    failed_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "failed_ccd_codes.json")
    with open(failed_file, "w") as f:
        json.dump([{"ccd_code": c, "reason": r} for c, r in failed], f, indent=2)
    print(f"Saved failed CCD codes with reasons to {failed_file}")


if __name__ == '__main__':
    main()
