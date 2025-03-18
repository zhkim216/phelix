import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import jax.numpy as jnp
import torch
from colabdesign import clear_mem
from colabdesign.af import mk_af_model
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding, EsmTokenizer

import omegafold
from allatom_design.data import data
from allatom_design.data.residue_constants import STANDARD_ATOM_MASK
from omegafold import pipeline as of_pipeline
from omegafold.utils.torch_utils import recursive_to
import argparse
from allatom_design.data import residue_constants as rc
import numpy as np
from colabdesign.af.alphafold.common import protein


def run_esmfold(sequence_list: List[str],
                residue_index: TensorType["b n", torch.long],
                model: EsmForProteinFolding,
                tokenizer: EsmTokenizer
                ) -> Dict[str, TensorType["b ..."]]:
    """
    Run ESMFold on a list of sequences.

    Returns a dict containing:
    - pred_coords: (b n 37 3) predicted coordinates of atoms
    - plddt: (b n 37) predicted pLDDTs for all atoms
    - ca_plddt: (b n) predicted pLDDTs for CA atoms
    - seq_mask: (b n) sequence mask
    - aatype: (b n) input amino acid types in AF2 format
    - atom_mask: (b n 37) atom mask corresponding to aatype
    - residue_index: (b n) residue index, usually just range(n)
    - avg_ca_plddt: (b) average CA pLDDT across sequence
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
    plddt = outputs.plddt * 100 * seq_mask[..., None]
    ca_plddt = plddt[:, :, 1]

    # Calculate average CA pLDDT
    avg_ca_plddt = (ca_plddt * seq_mask).sum(dim=-1) / seq_mask.sum(dim=-1).clamp(min=1e-3)

    esm_outputs = {
        "pred_coords": pred_coords_atom37,
        "plddt": plddt,
        "ca_plddt": ca_plddt,
        "seq_mask": seq_mask,
        "aatype": outputs.aatype,
        "residue_index": outputs.residue_index,
        "avg_ca_plddt": avg_ca_plddt,
    }
    esm_outputs = {k: v.cpu() for k, v in esm_outputs.items()}

    # Add atom mask based on input aatypes for convenience
    aatype, seq_mask = esm_outputs["aatype"], esm_outputs["seq_mask"]
    esm_outputs["atom_mask"] = torch.tensor(STANDARD_ATOM_MASK)[aatype] * seq_mask[..., None]

    return esm_outputs



def run_esmfold_batched(sequences_list: List[str],
                        residue_index_list: List[TensorType["n_s", int]],
                        chain_index_list: List[TensorType["n_s", int]],
                        model: EsmForProteinFolding,
                        tokenizer: EsmTokenizer,
                        max_tokens_per_batch: int = 1024,
                        ) -> Dict[str, List[TensorType["..."]]]:
    """
    Run ESMFold on a list of sequences, batching them by sequence length and to fit within a token limit.

    Returns a dict containing:
    - pred_coords: (b n 37 3) predicted coordinates of atoms
    - plddt: (b n 37) predicted pLDDTs
    - ca_plddt: (b n) predicted pLDDTs for CA atoms
    - seq_mask: (b n) sequence mask
    - aatype: (b n) input amino acid types in AF2 format
    - atom_mask: (b n 37) atom mask corresponding to aatype
    - residue_index: (b n) residue index, usually just range(n)
    - avg_ca_plddt: (b) average CA pLDDT across sequence
    """
    model = model.eval()
    esm_outputs = defaultdict(list)
    original_ids = []

    dataset = create_batched_seq_dataset(sequences_list, residue_index_list, chain_index_list, max_tokens_per_batch=max_tokens_per_batch)
    for batch in dataset:
        # Set up inputs
        inputs = tokenizer(
            batch["sequence"],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)

        # Add residue index gap of 1000 for chain separation
        residue_index = batch["residue_index"]
        residue_index = residue_index + (1000 * batch["chain_index"])

        inputs["position_ids"] = residue_index.to(model.device)

        # Run model
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process outputs
        seq_mask = inputs["attention_mask"]
        pred_coords_atom14 = outputs["positions"][-1]  # positions is shape (l, b, n, 14, 3)
        pred_coords_atom37 = data.atom14_aatype_to_atom37(pred_coords_atom14, outputs["aatype"])
        plddt = outputs["plddt"] * 100 * seq_mask[..., None]
        ca_plddt = plddt[:, :, 1]   # get pLDDT for CA atoms
        avg_ca_plddt = (ca_plddt * seq_mask).sum(dim=-1) / seq_mask.sum(dim=-1).clamp(min=1e-3)

        aatype, seq_mask = outputs.aatype.cpu(), seq_mask.cpu()
        atom_mask = torch.tensor(STANDARD_ATOM_MASK[aatype]) * seq_mask[..., None]

        # Create batch outputs
        esm_outputs_batch = {
            "pred_coords": pred_coords_atom37,
            "plddt": plddt,
            "ca_plddt": ca_plddt,
            "seq_mask": seq_mask,
            "aatype": aatype,
            "residue_index": residue_index,
            "chain_index": batch["chain_index"],
            "avg_ca_plddt": avg_ca_plddt[..., None],  # add sequence dimension for consistency
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
                               all_chain_indices: List[TensorType["n_s", int]],
                               max_tokens_per_batch: int = 1024,
                               ) -> Generator[dict, None, None]:
    """
    Create a batched dataset of sequences for ESMFold, sorting by sequence length and limiting batch size.

    Loosely based on https://github.com/facebookresearch/esm/blob/c9c7d4f0fec964ce10c3e11dccec6c16edaa5144/scripts/fold.py#L66
    """
    # Sort by sequence length
    B = len(all_sequences)
    examples = [(seq, residx, chain_idx, id) for seq, residx, chain_idx, id in zip(all_sequences, all_residue_indices, all_chain_indices, range(B))]
    examples = sorted(examples, key=lambda x: len(x[0]))

    # Define collator
    def collate_fn(examples: List[Tuple[str, TensorType["n", int], int]]) -> Dict[str, List]:
        """
        Given a list of examples, collate them into a batch with keys:
        - sequence: (b) sequence
        - residue_index: (b n) residue index
        - id: (b) unique identifier for each sequence
        """
        batch = {"sequence": [], "residue_index": [], "id": [], "chain_index": []}

        N = max(len(seq) for seq, _, _, _ in examples)
        for seq, residx, chain_idx, id in examples:
            batch["sequence"].append(seq)
            batch["residue_index"].append(data.make_fixed_size_1d(residx, fixed_size=N, start_idx=None))
            batch["chain_index"].append(data.make_fixed_size_1d(chain_idx, fixed_size=N, start_idx=None))
            batch["id"].append(id)

        batch["residue_index"] = torch.stack(batch["residue_index"], dim=0).to(torch.long)
        batch["chain_index"] = torch.stack(batch["chain_index"], dim=0).to(torch.long)
        return batch

    # Yield batches
    batch_examples, num_tokens = [], 0

    total_tokens = sum(len(seq) for seq in all_sequences)
    pbar = tqdm(total=total_tokens, desc="Number of ESMFold tokens processed", leave=False)

    for seq, residx, chain_idx, id in examples:
        # If adding this sequence would exceed the token limit, yield the current batch
        if num_tokens + len(seq) > max_tokens_per_batch and num_tokens > 0:
            yield collate_fn(batch_examples)
            batch_examples, num_tokens = [], 0

        # Add this sequence to the current batch
        batch_examples.append((seq, residx, chain_idx, id))
        num_tokens += len(seq)
        pbar.update(len(seq))

    yield collate_fn(batch_examples)

def save_best_model(af_model, filename):
    aux = af_model._tmp["best"]["aux"]["all"]
    plddt = np.mean(aux["plddt"], axis=-1)
    best_model_idx = np.argmax(plddt)
    p = {k:aux[k][best_model_idx] for k in ["aatype","residue_index","atom_positions","atom_mask"]}
    p["b_factors"] = 100 * p["atom_mask"] * aux["plddt"][best_model_idx][...,None]

    def to_pdb_str(x, n=None):
      p_str = protein.to_pdb(protein.Protein(**x))
      p_str = "\n".join(p_str.splitlines()[1:-2])
      return p_str

    p_str = to_pdb_str(p) + "\nEND\n"
    with open(filename, 'w') as f:
        f.write(p_str)


def run_af2(sequences_list: List[str],
            residue_index_list: List[TensorType["n_s", int]],
            chain_index_list: List[TensorType["n_s", int]],
            pdbs: List[str],  # used for extracting residue index. TODO remove dependence on pdb file
            af_model: mk_af_model,
            out_dir: str,
            num_models: int,
            sample_models: bool,
            num_recycles: int,
            save_best: bool = True,
            rm_template_interchain: bool = False,
            chains: Optional[str] = None,
            **kwargs) -> Tuple[Dict[str, torch.Tensor],
                               List[str]]:
    """
    Predict sequences with AlphaFold2.

    Return a tuple (dictionary of outputs, output filenames).
    """
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    output_files = []

    # Predict structures
    for _, (seq, pdb, residue_index, chain_index) in enumerate(zip(sequences_list, pdbs, residue_index_list, chain_index_list)):
        output_pdb = f"{out_dir}/af2_{Path(pdb).stem}.pdb"
        assert len(chain_index_list[0].unique()) == 1, "Multi-chain prediction not supported yet"
        # af_model.prep_inputs(pdb, chains, ignore_missing=False)
        _prep_struct_pred(af_model, residue_index)

        af_model.restart()
        af_model.set_opt("template", rm_ic=rm_template_interchain)
        af_model.predict(seq=seq,
                         num_models=num_models,
                         sample_models=sample_models,
                         num_recycles=num_recycles,
                         verbose=False)

        af_model._save_results(save_best=save_best, best_metric="plddt", verbose=False)

        if save_best:
            save_best_model(af_model, output_pdb)
        else:
            af_model.save_current_pdb(output_pdb)

        output_files.append(output_pdb)

    preds = [data.load_feats_from_pdb(pdb) for pdb in output_files]

    # Preprocess plddt-CA
    plddt = [pred["b_factors"] for pred in preds]
    ca_plddt = [pred["b_factors"][:, 1] for pred in preds]
    avg_ca_plddt = [torch.mean(ca_plddt, dim=0, keepdim=True) for ca_plddt in ca_plddt]  # keep sequence dim for consistency

    # Prepare AF2 outputs
    af2_outputs = {
        "pred_coords": [pred["all_atom_positions"] for pred in preds],
        "plddt": plddt,
        "ca_plddt": ca_plddt,
        "seq_mask": [pred["seq_mask"] for pred in preds],
        "aatype": [pred["aatype"] for pred in preds],
        "residue_index": [pred["residue_index"].long() for pred in preds],
        "avg_ca_plddt": avg_ca_plddt,
        "atom_mask": [pred["all_atom_mask"] for pred in preds],
    }

    return af2_outputs, output_files


def _prep_struct_pred(model: mk_af_model,
                      residue_index: TensorType["n_s", int]):
    '''
    Prep inputs for structure prediction without requiring an input PDB.
    Adapted from ColabDesign's _prep_fixbb function.
    ---------------------------------------------------
    if copies > 1:
      -homooligomer=True - input pdb chains are parsed as homo-oligomeric units
      -repeat=True       - tie the repeating sequence within single chain
    -rm_template_seq     - if template is defined, remove information about template sequence
    -fix_pos="1,2-10"    - specify which positions to keep fixed in the sequence
                           note: supervised loss is applied to all positions, use "partial"
                           protocol to apply supervised loss to only subset of positions
    -ignore_missing=True - skip positions that have missing density (no CA coordinate)
    ---------------------------------------------------
    '''
    # prep features
    residue_index = np.array(residue_index)
    model._len = residue_index.shape[0]
    model._lengths = [model._len]

    # feat dims
    num_seq = 1
    res_idx = residue_index

    # configure input features
    model._inputs = model._prep_features(num_res=sum(model._lengths), num_seq=num_seq)
    model._inputs["residue_index"] = res_idx
    model._wt_aatype = np.full(model._len, fill_value=-1, dtype=np.int64)

    model._prep_model()


def run_omegafold(sequences_list: List[str],
                  residue_index_list: List[TensorType["n_s", int]],
                  omegafold_model: omegafold.OmegaFold,
                  out_dir: str,
                  device: str,
                  num_pseudo_msa: int = 15,
                  mask_rate: float = 0.12,
                  num_recycle: int = 10,
                  deterministic: bool = True,
                  subbatch_size: Optional[int] = None,
                  **kwargs):
    raise NotImplementedError("OmegaFold not supported anymore; code needs to be updated")
    forward_config = argparse.Namespace(
        subbatch_size=subbatch_size,
        num_recycle=num_recycle,
    )
    of_inputs = omegafold_inputs(sequences_list, num_pseudo_msa, mask_rate, num_recycle, deterministic, device)
    of_outputs = defaultdict(list)

    for (of_input, of_aux), residue_index in tqdm(zip(of_inputs, residue_index_list), total=len(residue_index_list),
                                                  desc="Running OmegaFold", leave=False):
        # TODO: like ESMFold, omegafold seems to ignore discontiguous residues
        output = omegafold_model(of_input, predict_with_confidence=True, fwd_cfg=forward_config)

        # Save outputs
        aatype, seq_mask = of_aux["aatype"], of_aux["seq_mask"]
        atom37_coords = data.atom14_aatype_to_atom37(output["final_atom_positions"].detach().cpu(), aatype.detach().cpu())
        atom37_mask = torch.tensor(STANDARD_ATOM_MASK[aatype]) * seq_mask[..., None]

        of_outputs["pred_coords"].append(atom37_coords)
        of_outputs["plddt"].append(output["confidence"].detach().cpu())
        of_outputs["seq_mask"].append(seq_mask)
        of_outputs["aatype"].append(aatype)
        of_outputs["residue_index"].append(residue_index.detach().cpu())
        of_outputs["avg_ca_plddt"].append(output["confidence"].mean(dim=0, keepdim=True).detach().cpu())
        of_outputs["atom_mask"].append(atom37_mask)


    return of_outputs



def omegafold_inputs(sequences_list: List[str],
                     num_pseudo_msa: int,
                     mask_rate: float,
                     num_cycle: int,
                     deterministic: bool,
                     device: str):
    """
    Adapted from omegafold.pipeline.fasta2inputs
    """
    aux = {}

    for seq in sequences_list:
        seq = seq.replace("Z", "E").replace("B", "D").replace("U", "C")
        aatype = torch.LongTensor(
            [rc.restypes_with_x.index(aa) if aa != '-' else 21 for aa in seq]
        )
        mask = torch.ones_like(aatype).float()

        num_res = len(aatype)
        data = list()
        g = None
        if deterministic:
            g = torch.Generator()
            g.manual_seed(num_res)
        for _ in range(num_cycle):
            p_msa = aatype[None, :].repeat(num_pseudo_msa, 1)
            p_msa_mask = torch.rand([num_pseudo_msa, num_res], generator=g).gt(mask_rate)
            p_msa_mask = torch.cat((mask[None, :], p_msa_mask), dim=0)
            p_msa = torch.cat((aatype[None, :], p_msa), dim=0)
            p_msa[~p_msa_mask.bool()] = 21
            data.append({"p_msa": p_msa, "p_msa_mask": p_msa_mask})

        aux["aatype"] = aatype
        aux["seq_mask"] = mask
        yield recursive_to(data, device=device), aux


def get_omegafold_model(cache_file: str, device: str):
    Path(cache_file).parent.mkdir(exist_ok=True, parents=True)
    model = omegafold.OmegaFold(omegafold.make_config(1))
    state_dict = of_pipeline._load_weights("https://helixon.s3.amazonaws.com/release1.pt", cache_file)
    if "model" in state_dict:
        state_dict = state_dict.pop("model")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def get_esmfold_model(device: str):
    # Set up ESMFold
    esmfold = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()
    esmfold.esm = esmfold.esm.half()
    esmfold = esmfold.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    return esmfold, tokenizer


def get_struct_pred_model(cfg: DictConfig,
                          device: str) -> Dict[str, Any]:
    """
    Get structure prediction model components as a dictionary based on config.

    Example config:
    struct_pred_cfg:
        model_name: "esmfold"  # ["esmfold", "af2", "omegafold"]
        af2:
            data_dir: # directory containing "params/" with af2 model params
            num_models: 1  # only 1 model is currently supported
            sample_models: true  # randomly sample models from the ensemble
            num_recycles: 3
            use_multimer: false
        omegafold:
            cache_dir: # directory to cache omegafold model
    """
    model_name = cfg.model_name
    struct_pred_model = {"model_name": model_name, "cfg": cfg, "device": device}

    base_cfg = OmegaConf.load(cfg.base_cfg)
    cfg = OmegaConf.merge(base_cfg, cfg)

    if model_name == "af2":
        clear_mem()
        af_model = mk_af_model(data_dir=cfg.af2.data_dir,
                               use_multimer=cfg.af2.use_multimer)
        struct_pred_model["af_model"] = af_model
        af_model._get_loss = af_model._loss_unsupervised

    elif model_name == "esmfold":
        esmfold, tokenizer = get_esmfold_model(device=device)
        struct_pred_model["esmfold"] = esmfold
        struct_pred_model["tokenizer"] = tokenizer

    elif model_name == "omegafold":
        omegafold = get_omegafold_model(cache_file=cfg.omegafold.cache_file, device=device)
        struct_pred_model["omegafold"] = omegafold
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return struct_pred_model
