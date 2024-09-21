from collections import defaultdict
from typing import Dict, Generator, List, Tuple

import torch
from torchtyping import TensorType
from tqdm import tqdm
from transformers import EsmForProteinFolding, EsmTokenizer

from allatom_design.data import data
from allatom_design.data.residue_constants import STANDARD_ATOM_MASK


def run_esmfold(sequence_list: List[str],
                residue_index: TensorType["b n", torch.long],
                model: EsmForProteinFolding,
                tokenizer: EsmTokenizer
                ) -> Dict[str, TensorType["b ..."]]:
    """
    Run ESMFold on a list of sequences.

    Returns a dict containing:
    - pred_coords: (b n 37 3) predicted coordinates of atoms
    = plddts: (b n) predicted pLDDTs
    - seq_mask: (b n) sequence mask
    - aatype: (b n) input amino acid types in AF2 format
    - atom_mask: (b n 37) atom mask corresponding to aatype
    - residue_index: (b n) residue index, usually just range(n)
    - avg_plddt: (b) average pLDDT across sequence
    """
    model = model.eval()

    esm_outputs = {}

    # Set up inputs
    inputs = tokenizer(
        sequence_list,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)

    inputs["position_ids"] = residue_index

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process outputs
    seq_mask = inputs.attention_mask
    # positions is shape (l, b, n, 14, 3)
    pred_coords_atom14 = outputs.positions[-1]
    pred_coords_atom37 = data.atom14_aatype_to_atom37(pred_coords_atom14, outputs.aatype)
    plddts = outputs.plddt[:, :, 1] * seq_mask

    avg_plddt = (plddts * seq_mask).sum(dim=-1) / seq_mask.sum(dim=-1).clamp(min=1e-3)

    esm_outputs = {
        "pred_coords": pred_coords_atom37,
        "plddts": plddts,
        "seq_mask": seq_mask,
        "aatype": outputs.aatype,
        "residue_index": outputs.residue_index,
        "avg_plddt": avg_plddt,
    }
    esm_outputs = {k: v.cpu() for k, v in esm_outputs.items()}

    # Add atom mask based on input aatypes for convenience
    aatype, seq_mask = esm_outputs["aatype"], esm_outputs["seq_mask"]
    esm_outputs["atom_mask"] = torch.tensor(STANDARD_ATOM_MASK)[aatype] * seq_mask[..., None]

    return esm_outputs



def run_esmfold_batched(sequences_list: List[str],
                        residue_index_list: List[TensorType["n_s", int]],
                        model: EsmForProteinFolding,
                        tokenizer: EsmTokenizer,
                        max_tokens_per_batch: int = 1024,
                        ) -> Dict[str, List[TensorType["..."]]]:
    """
    Run ESMFold on a list of sequences, batching them by sequence length and to fit within a token limit.

    Returns a dict containing:
    - pred_coords: (b n 37 3) predicted coordinates of atoms
    - plddts: (b n) predicted pLDDTs
    - seq_mask: (b n) sequence mask
    - aatype: (b n) input amino acid types in AF2 format
    - atom_mask: (b n 37) atom mask corresponding to aatype
    - residue_index: (b n) residue index, usually just range(n)
    - avg_plddt: (b) average pLDDT across sequence
    """
    model = model.eval()
    esm_outputs = defaultdict(list)
    original_ids = []

    dataset = create_batched_seq_dataset(sequences_list, residue_index_list, max_tokens_per_batch=max_tokens_per_batch)
    for batch in dataset:
        # Set up inputs
        inputs = tokenizer(
            batch["sequence"],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)

        inputs["position_ids"] = batch["residue_index"].to(model.device)

        # Run model
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process outputs
        seq_mask = inputs["attention_mask"]
        pred_coords_atom14 = outputs["positions"][-1]  # positions is shape (l, b, n, 14, 3)
        pred_coords_atom37 = data.atom14_aatype_to_atom37(pred_coords_atom14, outputs["aatype"])
        plddts = outputs["plddt"][:, :, 1] * seq_mask
        avg_plddt = (plddts * seq_mask).sum(dim=-1) / seq_mask.sum(dim=-1).clamp(min=1e-3)

        aatype, seq_mask = outputs.aatype.cpu(), seq_mask.cpu()
        atom_mask = torch.tensor(STANDARD_ATOM_MASK[aatype]) * seq_mask[..., None]

        # Create batch outputs
        esm_outputs_batch = {
            "pred_coords": pred_coords_atom37,
            "plddts": plddts,
            "seq_mask": seq_mask,
            "aatype": aatype,
            "residue_index": batch["residue_index"],
            "avg_plddt": avg_plddt[..., None],  # add sequence dimension for consistency
            "atom_mask": atom_mask,    # add atom mask based on input aatypes for convenience
        }
        esm_outputs_batch = {k: v.cpu() for k, v in esm_outputs_batch.items()}

        # Crop each output to original sequence length
        seq_lens = seq_mask.sum(dim=-1).long()
        esm_outputs_batch = {k: [v[i, :l] for i, l in enumerate(seq_lens)] for k, v in esm_outputs_batch.items()}

        # Store outputs
        for k, v in esm_outputs_batch.items():
            esm_outputs[k].extend(v)

        # To preserve original sequence order
        original_ids.extend(batch["id"])

    # Reorder all outputs based on original sequence order
    reordered_outputs = {k: [] for k in esm_outputs}
    sort_indices = torch.argsort(torch.tensor(original_ids))
    for idx in sort_indices:
        for k in esm_outputs:
            reordered_outputs[k].append(esm_outputs[k][idx])

    return reordered_outputs



def create_batched_seq_dataset(all_sequences: List[str],
                               all_residue_indices: List[TensorType["n_s", int]],
                               max_tokens_per_batch: int = 1024,
                               ) -> Generator[dict, None, None]:
    """
    Create a batched dataset of sequences for ESMFold, sorting by sequence length and limiting batch size.

    Loosely based on https://github.com/facebookresearch/esm/blob/c9c7d4f0fec964ce10c3e11dccec6c16edaa5144/scripts/fold.py#L66
    """
    # Sort by sequence length
    B = len(all_sequences)
    examples = [(seq, residx, id) for seq, residx, id in zip(all_sequences, all_residue_indices, range(B))]
    examples = sorted(examples, key=lambda x: len(x[0]))

    # Define collator
    def collate_fn(examples: List[Tuple[str, TensorType["n", int], int]]) -> Dict[str, List]:
        """
        Given a list of examples, collate them into a batch with keys:
        - sequence: (b) sequence
        - residue_index: (b n) residue index
        - id: (b) unique identifier for each sequence
        """
        batch = {"sequence": [], "residue_index": [], "id": []}

        N = max(len(seq) for seq, _, _ in examples)
        for seq, residx, id in examples:
            batch["sequence"].append(seq)
            batch["residue_index"].append(data.make_fixed_size_1d(residx, fixed_size=N, start_idx=None))
            batch["id"].append(id)

        batch["residue_index"] = torch.stack(batch["residue_index"], dim=0).to(torch.long)
        return batch

    # Yield batches
    batch_examples, num_tokens = [], 0

    total_tokens = sum(len(seq) for seq in all_sequences)
    pbar = tqdm(total=total_tokens, desc="Number of ESMFold tokens processed", leave=False)

    for seq, residx, id in examples:
        # If adding this sequence would exceed the token limit, yield the current batch
        if num_tokens + len(seq) > max_tokens_per_batch and num_tokens > 0:
            yield collate_fn(batch_examples)
            batch_examples, num_tokens = [], 0

        # Add this sequence to the current batch
        batch_examples.append((seq, residx, id))
        num_tokens += len(seq)
        pbar.update(len(seq))

    yield collate_fn(batch_examples)
