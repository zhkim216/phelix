import copy

import torch
import torch.nn.functional as F
from torchtyping import TensorType

import allatom_design.data.const as const
from allatom_design.data.feature.feature_utils import unbatch_feats


def thread_sequence_onto_example(example: dict[str, TensorType["1 n ..."]],
                                 new_res_type: TensorType["n2 c", int],
                                 label_seq_id: TensorType["n2", int],
                                 mask: TensorType["n2", bool] | None = None) -> dict[str, TensorType["n ..."]]:
    """
    Thread a sequence onto an example. Returns a deepcopy of the example.

    - mask: if provided, only update residues where mask is True
    """
    example = copy.deepcopy(example)

    # Subset inputs based on mask, so that we only update the sequence where mask is True
    if mask is None:
        mask = torch.ones_like(label_seq_id, dtype=torch.bool)

    label_seq_id = label_seq_id[mask]
    new_res_type = new_res_type[mask]

    # For now, make sure there is only one protein chain
    protein_mask = example["mol_type"] == const.chain_type_ids["PROTEIN"]
    n_prot_chains = len(example["asym_id"][protein_mask].unique())
    if n_prot_chains > 1:
        raise ValueError(f"Found {n_prot_chains} protein chains in {example['record_id']}. For now, we only support threading sequences onto single-chain proteins.")

    # Set all missing protein residues to X and erase all sidechain coordinates
    protein_res_type = torch.full_like(example["mol_type"][protein_mask], const.token_ids["UNK"], dtype=torch.long)  # [1, n_prot_tokens]
    protein_res_type = F.one_hot(protein_res_type, num_classes=example["res_type"].shape[-1])  # [1, n_prot_tokens, 33]
    protein_res_type = protein_res_type.squeeze(0)  # temporarily squeeze out batch dimension
    protein_res_type[label_seq_id - 1] = new_res_type  # label_seq_id is 1-indexed

    example["res_type"][protein_mask] = protein_res_type.unsqueeze(0)
    example["coords"][example["prot_scn_atom_mask"].bool()] = 0.0
    example["atom_resolved_mask"][example["prot_scn_atom_mask"].bool()] = False

    example = unbatch_feats(example)[0]  # squeeze out batch dimension

    return example
