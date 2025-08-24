"""Allatom design constants."""
from collections.abc import Sequence
from functools import cached_property
from itertools import cycle

import biotite.structure as struc
import numpy as np
from atomworks.io.constants import (AA_LIKE_CHEM_TYPES, DNA_LIKE_CHEM_TYPES,
                                    GAP, RNA_LIKE_CHEM_TYPES, STANDARD_AA,
                                    STANDARD_DNA, STANDARD_RNA, UNKNOWN_AA,
                                    UNKNOWN_DNA, UNKNOWN_RNA)

"""Sequence tokens in AF3"""

# fmt: off
AF3_TOKENS = (
    # 20 AA + 1 unknown AA
    *STANDARD_AA, UNKNOWN_AA,
    # 4 RNA + 1 unknown RNA
    *STANDARD_RNA, UNKNOWN_RNA,
    # 4 DNA + 1 unknown DNA
    *STANDARD_DNA, UNKNOWN_DNA,
    # 1 gap
    GAP,
)
# fmt: on

class AF3SequenceEncoding:
    """
    Encodes and decodes sequence tokens for AlphaFold 3.

    This class provides functionality to convert between residue names and their
    corresponding integer encodings as used in AlphaFold 3. It handles standard
    amino acids, RNA, DNA, and unknown residues.

    Methods:
        encode(res_names): Encode residue names to integer indices.
        decode(res_indices): Decode integer indices to residue names.
        tokens: Property that returns the list of AF3 tokens.
        n_tokens: Property that returns the number of AF3 tokens.
    """

    def __init__(self):
        # Load CCD from biotite
        ccd = struc.info.ccd.get_ccd()

        # Get all residue names and their corresponding chemtypes
        self.all_res_names = ccd["chem_comp"]["id"].as_array()
        self.all_res_chemtypes = np.char.upper(ccd["chem_comp"]["type"].as_array())

        # Get boolean arrays for each chemtype

        self.is_rna_like = np.isin(self.all_res_chemtypes, list(RNA_LIKE_CHEM_TYPES))
        self.is_dna_like = np.isin(self.all_res_chemtypes, list(DNA_LIKE_CHEM_TYPES))
        self.is_aa_like = np.isin(self.all_res_chemtypes, list(AA_LIKE_CHEM_TYPES))

        # Build mappings for all CCD residue names to AF3 tokens
        res_name_to_token = dict(zip(self.all_res_names[self.is_rna_like], cycle([UNKNOWN_RNA])))
        res_name_to_token |= dict(zip(self.all_res_names[self.is_dna_like], cycle([UNKNOWN_DNA])))
        res_name_to_token |= dict(zip(AF3_TOKENS, AF3_TOKENS, strict=False))
        self.res_name_to_token = res_name_to_token

        # Build mappings for AF3 tokens to indices
        self.af3_token_to_int = {token: i for i, token in enumerate(AF3_TOKENS)}

    @property
    def tokens(self) -> list[str]:
        return AF3_TOKENS

    def res_name_to_af3_token(self, res_name: str) -> str:
        return np.vectorize(lambda res_name: self.res_name_to_token.get(res_name, UNKNOWN_AA))(res_name)

    @property
    def token_to_idx(self) -> dict[str, int]:
        return self.af3_token_to_int

    @cached_property
    def idx_to_token(self) -> np.ndarray:
        return np.array(AF3_TOKENS)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def encode(self, res_names: Sequence[str]) -> np.ndarray:
        # NOTE: Defined here rather than as attribute to allow pickling for multiprocessing
        encode_func = np.vectorize(lambda x: self.af3_token_to_int.get(x, self.af3_token_to_int[UNKNOWN_AA]))
        return encode_func(res_names)

    def decode(self, token_idxs: int | Sequence[int]) -> np.ndarray:
        return self.idx_to_token[token_idxs]


AF3_SEQUENCE_ENCODING = AF3SequenceEncoding()


max_num_atoms = 23
