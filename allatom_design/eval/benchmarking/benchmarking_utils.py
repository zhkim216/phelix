import copy

import torch
import torch.nn.functional as F
from torchtyping import TensorType

import allatom_design.data.const as const


def thread_sequence_onto_example(example: dict[str, TensorType["1 n ..."]],
                                 res_type_one_hot: TensorType["n c", int],
                                 label_seq_id: TensorType["n", int],
                                 ) -> dict[str, TensorType["1 n ..."]]:
    """
    Thread a sequence onto an example. Returns a deepcopy of the example.
    """
    example = copy.deepcopy(example)

    # For now, make sure there is only one protein chain
    protein_mask = example["mol_type"] == const.chain_type_ids["PROTEIN"]
    n_prot_chains = len(example["asym_id"][protein_mask].unique())
    if n_prot_chains > 1:
        raise ValueError(f"Found {n_prot_chains} protein chains in {example['record_id']}. For now, we only support threading sequences onto single-chain proteins.")

    # Set all protein residues to X and erase all sidechain coordinates
    protein_res_type = torch.full_like(protein_mask, const.token_ids["UNK"], dtype=torch.long)
    protein_res_type = F.one_hot(protein_res_type, num_classes=example["res_type"].shape[-1])  # [1, n_protein, 33]
    protein_res_type = protein_res_type.squeeze(0)  # temporarily squeeze out batch dimension
    protein_res_type[label_seq_id - 1] = res_type_one_hot  # label_seq_id is 1-indexed
    example["res_type"][protein_mask] = protein_res_type.unsqueeze(0)
    example["coords"][example["prot_scn_atom_mask"].bool()] = 0.0

    return example
