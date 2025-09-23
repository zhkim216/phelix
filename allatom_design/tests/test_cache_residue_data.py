import biotite.structure as struc
from atomworks.ml.transforms.cached_residue_data import load_cached_residue_level_data
from atomworks.io.utils.ccd import atom_array_from_ccd_code
from atomworks.ml.transforms.atom_array import AddGlobalResIdAnnotation
from atomworks.ml.transforms.cached_residue_data import LoadCachedResidueLevelData, RandomSubsampleCachedConformers
from atomworks.ml.transforms.base import Compose
from atomworks.io.tools.rdkit import atom_array_to_rdkit, atom_array_from_rdkit
import numpy as np

def test_load_cached_residue_level_data(ccd_code: str= "ALA",
                                        cache_dir = "/home/possu/jinho/allatom-design/atomworks_test/250922/residue_cache_data"):
    aa = atom_array_from_ccd_code(ccd_code)
    result = load_cached_residue_level_data(
        aa,
        dir=cache_dir,
        sharding_depth=1,            # 로더 기본값과 일치
        file_extension=".pt",        # 저장 확장자와 일치   
    )

    assert "residues" in result and ccd_code in result["residues"], f"{ccd_code} not found in result"
    assert "mol" in result["residues"][ccd_code], "mol not found in result"
    assert "descriptors" in result["residues"][ccd_code], "descriptors not found in result"
    assert "atom_names" in result["residues"][ccd_code], "atom_names not found in result"

    print(result["residues"][ccd_code]["mol"])
    print(result["residues"][ccd_code]["descriptors"])
    print(result["residues"][ccd_code]["fingerprint"])
    print(result["residues"][ccd_code]["atom_names"])

def test_transforms(ccd_code: str= "ALA",
                    cache_dir = "/home/possu/jinho/allatom-design/atomworks_test/250922/residue_cache_data"):
    
    n_conformers_to_sample = 2
    pipeline = Compose([
    AddGlobalResIdAnnotation(),
    LoadCachedResidueLevelData(
        dir=cache_dir,
        sharding_depth=1,
        file_extension=".pt",
    ),
    RandomSubsampleCachedConformers(n_conformers=n_conformers_to_sample),
])
    
    atom_array = atom_array_from_ccd_code(ccd_code)
    mol = atom_array_to_rdkit(atom_array, hydrogen_policy="remove")
    atom_array = atom_array_from_rdkit(mol)
    data = {"atom_array": atom_array}
    data = pipeline(data)
    
    assert "residues" in data["cached_residue_level_data"]
    assert "residue_conformer_indices" in data
    indices_dict = data["residue_conformer_indices"]

    atom_array = data["atom_array"]
    for global_res_id, conformer_indices in indices_dict.items():
        assert isinstance(conformer_indices, np.ndarray)
        assert len(conformer_indices) == n_conformers_to_sample

        # Verify that the conformer indices are within bounds
        res_mask = atom_array.res_id_global == global_res_id
        res_name = atom_array.res_name[res_mask][0]
        res_data = data["cached_residue_level_data"]["residues"][res_name]
        n_available = res_data["mol"].GetNumConformers()
        assert all(0 <= idx < n_available for idx in conformer_indices)

    print(f"{ccd_code} test_transforms done, success")

    return

if __name__ == "__main__":
    test_transforms()