"""
Convert between OpenFold and Boltz representations.
"""
from allatom_design.data import residue_constants as rc
from allatom_design.data import const

for resname, atom14_names in rc.restype_name_to_atom14_names.items():
    if resname == "UNK":
        continue
    atom14_names_unpadded = [x for x in atom14_names if x != ""]
    assert const.ref_atoms[resname] == atom14_names_unpadded, f"Discrepancy between {resname} and {const.ref_atoms[resname]}"


# === Conversion between boltz token id to openfold restype id === #
def _boltz_token_id_to_restype_id(token_id: int) -> int:
    token = const.tokens[token_id]  # boltz aa token is in restype_3
    restype_1 = rc.restype_3to1.get(token, "X")  # any unrecognized token is mapped to "X" (includes dna/rna/ligands)
    return rc.restype_order_with_x[restype_1]

boltz_token_id_to_restype_id = {token_id: _boltz_token_id_to_restype_id(token_id) for token_id in range(const.num_tokens)}
