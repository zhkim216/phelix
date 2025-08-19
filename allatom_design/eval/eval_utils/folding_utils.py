from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import hydra
import torch
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.types import Manifest
from boltz.model.model import Boltz1
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torchtyping import TensorType
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding, EsmTokenizer

from allatom_design.data import data
from allatom_design.data.residue_constants import STANDARD_ATOM_MASK
from allatom_design.data.write.writer import BoltzWriter

try:
    from colabdesign import clear_mem, mk_afdesign_model
    from colabdesign.af import mk_af_model
except ImportError:
    print("ColabDesign not installed, skipping import")


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
        "pred_coords_atom14": pred_coords_atom14,
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
            "pred_coords_atom14": pred_coords_atom14,
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


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@dataclass
class BoltzSteeringParams:
    """Steering parameters."""

    fk_steering: bool = True
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    guidance_update: bool = True
    num_gd_steps: int = 16


@dataclass
class PairformerArgs:
    """Pairformer arguments."""

    num_blocks: int = 48
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    use_trifast: bool = True


@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    use_trifast: bool = True



def make_boltz_trainer(processed_data_dir: str,
                       out_dir: str,
                       num_workers: int):
    processed_data_dir = Path(processed_data_dir)
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_data_dir / "manifest_unclustered.json"),
        targets_dir=processed_data_dir / "structures",
        msa_dir=processed_data_dir / "msa",
    )

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
    )

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format="mmcif",
    )

    trainer = Trainer(
        default_root_dir=out_dir,
        strategy="auto",
        callbacks=[pred_writer],
        accelerator="gpu",
        devices=1,
        precision=32,
    )

    return trainer, data_module



def run_af2(sequences_list: list[str],
            residue_index_list: list[TensorType["n_s", int]],
            chain_index_list: list[TensorType["n_s", int]],
            pdbs: list[str],  # used for extracting residue index. TODO remove dependence on pdb file
            af_model: "mk_af_model",
            out_dir: str,
            num_models: int,
            sample_models: bool,
            num_recycles: int,
            save_best: bool = True,
            rm_template_interchain: bool = False,
            chains: str | None = None,
            **kwargs) -> tuple[dict[str, torch.Tensor], list[str]]:
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



def get_boltz_model(boltz_cfg: DictConfig, device: str) -> Boltz1:
    diffusion_params = BoltzDiffusionParams()
    steering_args = BoltzSteeringParams()
    pairformer_args = PairformerArgs(use_trifast=True)
    msa_module_args = MSAModuleArgs(use_trifast=True)

    model_module: Boltz1 = Boltz1.load_from_checkpoint(boltz_cfg["checkpoint"],
                                                       strict=True,
                                                       predict_args=boltz_cfg["predict_args"],
                                                       map_location="cpu",
                                                       diffusion_process_args=asdict(diffusion_params),
                                                       ema=False,
                                                       pairformer_args=asdict(pairformer_args),
                                                       msa_module_args=asdict(msa_module_args),
                                                       steering_args=asdict(steering_args))
    model_module = model_module.eval()
    return model_module


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
        model_name: "esmfold"  # ["esmfold", "boltz1"]
        boltz1:
        esmfold:
            max_tokens_per_batch: 1024
        af2_interface:
            data_dir: # directory containing "params/" with af2 model params
            num_models: 1
            num_recycles: 3
            use_multimer: false
    """
    model_name = cfg.model_name
    base_cfg = OmegaConf.load(cfg.base_cfg)
    cfg = OmegaConf.merge(base_cfg, cfg)

    struct_pred_model = {"model_name": model_name, "cfg": cfg, "device": device}
    if model_name == "boltz1":
        struct_pred_model["boltz1"] = get_boltz_model(cfg.boltz1, device=device)
        struct_pred_model["trainer_fn"] = partial(make_boltz_trainer,
                                                  num_workers=cfg.boltz1.num_workers)
        struct_pred_model["data_cfg"] = hydra.utils.instantiate(cfg.boltz1.data_cfg)

    elif model_name == "esmfold":
        esmfold, tokenizer = get_esmfold_model(device=device)
        struct_pred_model["esmfold"] = esmfold
        struct_pred_model["tokenizer"] = tokenizer
        struct_pred_model["data_cfg"] = hydra.utils.instantiate(cfg.boltz1.data_cfg)  # useful to have boltz tokenizer/featurizer
    elif model_name == "af2_interface":
        clear_mem()
        af2_cfg = cfg.af2_interface

        # get AF2 model for predicting complex
        if af2_cfg.hard_target:
            complex_prediction_model = mk_afdesign_model(protocol="binder", num_recycles=af2_cfg.num_recycles, data_dir=af2_cfg.data_dir,
                                                         use_multimer=False, use_initial_guess=True, use_initial_atom_pos=False)
        else:
            complex_prediction_model = mk_afdesign_model(protocol="binder", num_recycles=af2_cfg.num_recycles, data_dir=af2_cfg.data_dir,
                                                         use_multimer=False, use_initial_guess=False, use_initial_atom_pos=False)

        # get AF2 model for predicting binder in isolation
        af_model = mk_af_model(use_multimer=False,
                               use_templates=False,
                               best_metric="ptm",
                               data_dir=af2_cfg.data_dir)

        struct_pred_model["af_model_complex"] = complex_prediction_model
        struct_pred_model["af_model_binder"] = af_model
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return struct_pred_model
