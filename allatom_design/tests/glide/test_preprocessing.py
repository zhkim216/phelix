"""Tests for Glide preprocessing module."""

from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray

from allatom_design.eval.glide.preprocessing import (
    compute_ligand_centroid,
    get_ligand_pn_unit_iids,
    get_protein_pn_unit_iids,
    preprocess_structure,
    write_ligand_sdf,
)

from allatom_design.tests.glide.conftest import (
    EXAMPLE_CIF,
    requires_example_data,
)


# ============================================================================
# Unit tests (no external data needed)
# ============================================================================


class TestGetChainIds:
    """Test protein/ligand chain ID extraction."""

    def _make_atom_array(self, n_prot: int = 10, n_lig: int = 5) -> AtomArray:
        """Create a minimal atom array with protein and ligand atoms."""
        import atomworks.enums as aw_enums

        n = n_prot + n_lig
        arr = AtomArray(n)
        arr.coord = np.random.default_rng(42).standard_normal((n, 3))
        arr.element = np.array(["C"] * n)
        arr.atom_name = np.array(["CA"] * n_prot + ["C1"] * n_lig)
        arr.res_name = np.array(["ALA"] * n_prot + ["LIG"] * n_lig)

        chain_type = np.zeros(n, dtype=int)
        chain_type[:n_prot] = int(aw_enums.ChainType.POLYPEPTIDE_L)
        chain_type[n_prot:] = int(aw_enums.ChainType.NON_POLYMER)
        arr.set_annotation("chain_type", chain_type)

        pn_unit_iid = np.array(["A_1"] * n_prot + ["B_1"] * n_lig)
        arr.set_annotation("pn_unit_iid", pn_unit_iid)

        return arr

    def test_get_protein_ids(self):
        arr = self._make_atom_array()
        ids = get_protein_pn_unit_iids(arr)
        assert ids == ["A_1"]

    def test_get_ligand_ids(self):
        arr = self._make_atom_array()
        ids = get_ligand_pn_unit_iids(arr)
        assert ids == ["B_1"]

    def test_empty_protein(self):
        arr = self._make_atom_array(n_prot=0, n_lig=5)
        ids = get_protein_pn_unit_iids(arr)
        assert ids == []

    def test_empty_ligand(self):
        arr = self._make_atom_array(n_prot=10, n_lig=0)
        ids = get_ligand_pn_unit_iids(arr)
        assert ids == []

    def test_multiple_chains(self):
        import atomworks.enums as aw_enums

        n = 20
        arr = AtomArray(n)
        arr.coord = np.zeros((n, 3))
        arr.element = np.array(["C"] * n)
        arr.atom_name = np.array(["CA"] * n)
        arr.res_name = np.array(["ALA"] * 10 + ["LIG"] * 5 + ["LIG2"] * 5)

        chain_type = np.zeros(n, dtype=int)
        chain_type[:10] = int(aw_enums.ChainType.POLYPEPTIDE_L)
        chain_type[10:] = int(aw_enums.ChainType.NON_POLYMER)
        arr.set_annotation("chain_type", chain_type)

        pn_unit_iid = np.array(
            ["A_1"] * 5 + ["A_2"] * 5 + ["B_1"] * 5 + ["C_1"] * 5
        )
        arr.set_annotation("pn_unit_iid", pn_unit_iid)

        prot_ids = get_protein_pn_unit_iids(arr)
        lig_ids = get_ligand_pn_unit_iids(arr)
        assert prot_ids == ["A_1", "A_2"]
        assert lig_ids == ["B_1", "C_1"]

    def test_excludes_single_metal_ions(self):
        """Single metal ions (e.g. Zn, Fe) should be excluded from ligand detection."""
        import atomworks.enums as aw_enums

        # 10 protein atoms + 5 ligand atoms + 1 Zn ion + 1 Fe ion
        n = 17
        arr = AtomArray(n)
        arr.coord = np.zeros((n, 3))
        arr.element = np.array(
            ["C"] * 10 + ["C"] * 5 + ["ZN"] + ["FE"]
        )
        arr.atom_name = np.array(
            ["CA"] * 10 + ["C1"] * 5 + ["ZN"] + ["FE"]
        )
        arr.res_name = np.array(
            ["ALA"] * 10 + ["LIG"] * 5 + ["ZN"] + ["FE"]
        )

        chain_type = np.zeros(n, dtype=int)
        chain_type[:10] = int(aw_enums.ChainType.POLYPEPTIDE_L)
        chain_type[10:] = int(aw_enums.ChainType.NON_POLYMER)
        arr.set_annotation("chain_type", chain_type)

        pn_unit_iid = np.array(
            ["A_1"] * 10 + ["B_1"] * 5 + ["C_1"] + ["D_1"]
        )
        arr.set_annotation("pn_unit_iid", pn_unit_iid)

        lig_ids = get_ligand_pn_unit_iids(arr)
        # B_1 (real ligand) included; C_1 (Zn) and D_1 (Fe) excluded
        assert lig_ids == ["B_1"]

    def test_multi_atom_metal_complex_not_excluded(self):
        """A multi-atom non-polymer containing metals should NOT be excluded."""
        import atomworks.enums as aw_enums

        # 5 protein + 3-atom metal complex (e.g. heme fragment)
        n = 8
        arr = AtomArray(n)
        arr.coord = np.zeros((n, 3))
        arr.element = np.array(["C"] * 5 + ["FE", "N", "C"])
        arr.atom_name = np.array(["CA"] * 5 + ["FE", "NA", "C1"])
        arr.res_name = np.array(["ALA"] * 5 + ["HEM"] * 3)

        chain_type = np.zeros(n, dtype=int)
        chain_type[:5] = int(aw_enums.ChainType.POLYPEPTIDE_L)
        chain_type[5:] = int(aw_enums.ChainType.NON_POLYMER)
        arr.set_annotation("chain_type", chain_type)

        pn_unit_iid = np.array(["A_1"] * 5 + ["B_1"] * 3)
        arr.set_annotation("pn_unit_iid", pn_unit_iid)

        lig_ids = get_ligand_pn_unit_iids(arr)
        assert lig_ids == ["B_1"]


class TestComputeLigandCentroid:
    """Test ligand centroid computation."""

    def test_centroid_basic(self):
        arr = AtomArray(4)
        arr.coord = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]], dtype=float)
        arr.element = np.array(["C", "C", "C", "C"])
        centroid = compute_ligand_centroid(arr)
        np.testing.assert_allclose(centroid, [1.0, 1.0, 0.0])

    def test_centroid_excludes_hydrogen(self):
        arr = AtomArray(3)
        arr.coord = np.array([[0, 0, 0], [2, 0, 0], [100, 100, 100]], dtype=float)
        arr.element = np.array(["C", "C", "H"])
        centroid = compute_ligand_centroid(arr)
        np.testing.assert_allclose(centroid, [1.0, 0.0, 0.0])

    def test_centroid_all_hydrogen(self):
        arr = AtomArray(2)
        arr.coord = np.array([[1, 1, 1], [3, 3, 3]], dtype=float)
        arr.element = np.array(["H", "H"])
        centroid = compute_ligand_centroid(arr)
        np.testing.assert_allclose(centroid, [2.0, 2.0, 2.0])


class TestWriteLigandSdf:
    """Test ligand SDF writing via RDKit."""

    def test_write_sdf(self, tmp_path):
        """Test SDF writing with a simple molecule built from SMILES."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from atomworks.io.tools.rdkit import atom_array_from_rdkit

        mol = Chem.MolFromSmiles("CCO")  # ethanol
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = Chem.RemoveHs(mol)

        # Convert to atom array via atomworks roundtrip
        arr = atom_array_from_rdkit(mol)

        sdf_path = str(tmp_path / "test.sdf")
        write_ligand_sdf(arr, sdf_path)

        assert Path(sdf_path).exists()
        assert Path(sdf_path).stat().st_size > 0

        # Verify we can read it back
        read_mol = next(Chem.SDMolSupplier(sdf_path, removeHs=True))
        assert read_mol is not None
        assert read_mol.GetNumAtoms() > 0


# ============================================================================
# Integration tests (requires example CIF data)
# ============================================================================


@requires_example_data
class TestPreprocessStructure:
    """Integration tests using actual AF3 predicted CIF."""

    def test_preprocess_basic(self, tmp_work_dir):
        result = preprocess_structure(
            cif_path=EXAMPLE_CIF,
            out_dir=tmp_work_dir,
        )

        # Check returned keys
        assert "sample_id" in result
        assert "protein_pdb_path" in result
        assert "ligand_sdf_path" in result
        assert "ligand_centroid" in result
        assert "atom_array" in result
        assert "receptor_pn_unit_iids" in result
        assert "ligand_pn_unit_iids" in result

        # Check files were created
        assert Path(result["protein_pdb_path"]).exists()
        assert Path(result["ligand_sdf_path"]).exists()

        # Check PDB file has content
        pdb_content = Path(result["protein_pdb_path"]).read_text()
        assert "ATOM" in pdb_content

        # Check SDF file has content
        sdf_content = Path(result["ligand_sdf_path"]).read_text()
        assert len(sdf_content) > 0

        # Check centroid is a 3D point
        assert result["ligand_centroid"].shape == (3,)

        # Check chain IDs were detected
        assert len(result["receptor_pn_unit_iids"]) > 0
        assert len(result["ligand_pn_unit_iids"]) > 0

    def test_preprocess_custom_chains(self, tmp_work_dir):
        result = preprocess_structure(
            cif_path=EXAMPLE_CIF,
            out_dir=tmp_work_dir,
            receptor_pn_unit_iids=["A_1"],
            ligand_pn_unit_iids=["D_1"],
        )
        assert result["receptor_pn_unit_iids"] == ["A_1"]
        assert result["ligand_pn_unit_iids"] == ["D_1"]

    def test_preprocess_custom_sample_id(self, tmp_work_dir):
        result = preprocess_structure(
            cif_path=EXAMPLE_CIF,
            out_dir=tmp_work_dir,
            sample_id="custom_name",
        )
        assert result["sample_id"] == "custom_name"
        assert "custom_name" in result["protein_pdb_path"]
