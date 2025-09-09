import atomworks.ml.preprocessing.utils.structure_utils as dp
from tests.ml.preprocessing.test_prior_bugs_and_edge_cases import EDGE_CASE_LIST


def test_ligand_validity_retrieval():
    found_any = False
    for pdb_id in EDGE_CASE_LIST:
        # Check that ligand validity scores are properly retrieved
        ligand_validity_scores = dp.get_ligand_validity_scores_from_pdb_id(pdb_id)
        if len(ligand_validity_scores) > 0:
            found_any = True
    assert found_any, "No ligand validity scores found for any PDB ID."
