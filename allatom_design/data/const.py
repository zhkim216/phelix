"""Allatom design constants."""
from collections.abc import Sequence
from functools import cached_property
from itertools import cycle
from typing import Final
import torch

import biotite.structure as struc
import numpy as np
from atomworks.constants import (AA_LIKE_CHEM_TYPES, DICT_THREE_TO_ONE,
                                    DNA_LIKE_CHEM_TYPES, GAP,
                                    RNA_LIKE_CHEM_TYPES, STANDARD_AA,
                                    STANDARD_DNA, STANDARD_RNA, UNKNOWN_AA,
                                    UNKNOWN_DNA, UNKNOWN_RNA)
from atomworks.io.utils import sequence as aw_sequence

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

    @property
    def protein_tokens(self) -> list[str]:
        return [token for token in self.tokens if token in PROT_LETTER_TO_TOKEN.values()]

    @property
    def non_protein_tokens(self) -> list[str]:
        return [token for token in self.tokens if token not in PROT_LETTER_TO_TOKEN.values()]

    def encode(self, res_names: Sequence[str]) -> list[int]:
        return [self.af3_token_to_int.get(x, self.af3_token_to_int[UNKNOWN_AA]) for x in res_names]

    def decode(self, token_idxs: int | Sequence[int]) -> list[str]:
        if isinstance(token_idxs, int):
            token_idxs = [token_idxs]
        return [self.idx_to_token[idx] for idx in token_idxs]

    def encode_aa(self, aa: str) -> int:
        """First converts 1-letter AA name to protein token, then encodes to integer index."""
        return self.af3_token_to_int.get(PROT_LETTER_TO_TOKEN[aa], self.af3_token_to_int[UNKNOWN_AA])

    def encode_aa_seq(self, aa_seq: Sequence[str]) -> list[int]:
        """First converts 1-letter AA names to protein tokens, then encodes to integer indices."""
        return [self.encode_aa(aa) for aa in aa_seq]

    def decode_aa_seq(self, token_idxs: Sequence[int]) -> str:
        """First converts integer indices to tokens, then decodes to a string."""
        return "".join([PROT_TOKEN_TO_LETTER[token] for token in self.decode(token_idxs)])


AF3_ENCODING: Final[AF3SequenceEncoding] = AF3SequenceEncoding()

MAX_NUM_ATOMS: Final[int] = 23
PROT_BB_ATOMS: Final[list[str]] = ["N", "CA", "C", "O"]
PROT_LETTER_TO_TOKEN: Final[dict[str, str]] = {**aw_sequence.aa_chem_comp_1to3(), "X": "UNK"}  # include "X" for unknown amino acids
PROT_TOKEN_TO_LETTER: Final[dict[str, str]] = {v: k for k, v in PROT_LETTER_TO_TOKEN.items()}

DUMMY_SEQ_ID: Final[int] = -1  # dummy sequence id to use for auth_seq_id when not present

# Adapted from BioLip2, by JH
VDW_DICT = {
    "AC": 2.00, "AG": 1.72, "AL": 2.00, "AM": 2.00, "AR": 1.88, "AS": 1.85, "AT": 2.00, "AU": 1.66,
    "B": 2.00, "BA": 2.00, "BE": 2.00, "BH": 2.00, "BI": 2.00, "BK": 2.00, "BR": 1.85, "C": 1.70,
    "CA": 2.00, "CD": 1.58, "CE": 2.00, "CF": 2.00, "CL": 1.75, "CM": 2.00, "CO": 2.00, "CR": 2.00,
    "CS": 2.00, "CU": 1.40, "DB": 2.00, "DS": 2.00, "DY": 2.00, "ER": 2.00, "ES": 2.00, "EU": 2.00,
    "F": 1.47, "FE": 2.00, "FM": 2.00, "FR": 2.00, "GA": 1.87, "H": 1.09, "GD": 2.00, "GE": 2.00,
    "HE": 1.40, "HF": 2.00, "HG": 1.55, "HO": 2.00, "HS": 2.00, "I": 1.98, "K": 2.75, "IN": 1.93,
    "IR": 2.00, "KR": 2.02, "LA": 2.00, "LI": 1.82, "LR": 1.50, "LU": 2.00, "MD": 2.00, "MG": 1.73,
    "MN": 2.00, "MO": 2.00, "MT": 2.00, "N": 1.55, "O": 1.52, "NA": 2.27, "NB": 2.00, "ND": 2.00,
    "NE": 1.54, "NI": 1.63, "NO": 2.00, "NP": 2.00, "OS": 2.00, "P": 1.80, "S": 1.80, "PA": 2.00,
    "PB": 2.02, "PD": 1.63, "PM": 2.00, "PO": 2.00, "PR": 2.00, "PT": 1.72, "PU": 2.00, "RA": 2.00,
    "RB": 2.00, "RE": 2.00, "RF": 2.00, "RH": 2.00, "RN": 2.00, "RU": 2.00, "SB": 2.00, "SC": 2.00,
    "SE": 1.90, "SG": 2.00, "SI": 2.10, "SM": 2.00, "SN": 2.17, "SR": 2.00, "TA": 2.00, "TB": 2.00,
    "TC": 2.00, "TE": 2.06, "TH": 2.00, "TI": 2.00, "TL": 1.96, "TM": 2.00, "U": 1.86, "V": 2.00,
    "W": 2.00, "XE": 2.16, "YB": 2.00, "ZN": 1.39, "ZR": 2.00, "Y": 2.00, "D": 1.00, "XD": 1.09
}

# adapted from LigandMPNN
PERIODIC_TABLE_FEATURES: Final[torch.Tensor] = [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                    63,
                    64,
                    65,
                    66,
                    67,
                    68,
                    69,
                    70,
                    71,
                    72,
                    73,
                    74,
                    75,
                    76,
                    77,
                    78,
                    79,
                    80,
                    81,
                    82,
                    83,
                    84,
                    85,
                    86,
                    87,
                    88,
                    89,
                    90,
                    91,
                    92,
                    93,
                    94,
                    95,
                    96,
                    97,
                    98,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                ],
                [
                    0,
                    1,
                    18,
                    1,
                    2,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                ],
                [
                    0,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                ],
            ]