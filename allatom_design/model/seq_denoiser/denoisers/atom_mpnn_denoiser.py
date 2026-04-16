import copy
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from atomworks.ml.utils.token import apply_token_wise, spread_token_wise
from biotite.structure import AtomArray
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.const as const
import allatom_design.model.seq_denoiser.denoisers.seq_design.potts as potts
from allatom_design.data.data import to
from allatom_design.utils.feature_utils import slice_feats
from allatom_design.eval.eval_utils.sampling_utils import (
    get_timesteps_from_schedule, get_decoding_order,
)
from allatom_design.model.seq_denoiser.denoisers.denoiser import \
    BaseSeqDenoiser
from allatom_design.model.seq_denoiser.denoisers.seq_design.atom_mpnn import \
    AtomMPNN
from chroma.layers import complexity


class AtomMPNNDenoiser(BaseSeqDenoiser):
    def __init__(self,
                 cfg: DictConfig,
                 sigma_data: tuple[TensorType[(), float], TensorType[(), float]]):
        super().__init__()

        self.cfg = cfg
        self.bb_sigma_data, self.scn_sigma_data = sigma_data
        self.task = cfg.task

        # Sequence design model: AtomMPNN
        self.atom_mpnn = AtomMPNN(cfg.mpnn)


    def forward(self,
                batch: dict[str, TensorType["b ..."]],
                is_sampling: bool = False,
                sampling_inputs: dict[str, Any] | None = None,
                ) -> tuple[TensorType["b n c", float],  # seq_logits
                           dict[str, TensorType["b ..."]]]:
        # Build some helpful masks based on conditioning sequence and atoms
        batch = self.build_masks(batch, is_sampling)
        
        # Run model
        seq_logits, mpnn_feats = self.atom_mpnn(batch, is_sampling)

        # Outputs
        aux_preds = {
            "seq_logits": seq_logits,
            "potts_decoder_aux": mpnn_feats.get("potts_decoder_aux", None),
            "seq_cond_mask": batch["seq_cond_mask"],
            "atom_cond_mask": batch["atom_cond_mask"],
            "token_exists_mask": batch["token_exists_mask"],
            "protein_residue_node_mask": batch["protein_residue_node_mask"],
        }        

        return seq_logits, aux_preds


    def build_masks(self, batch: dict[str, TensorType["b ..."]], is_sampling) -> dict[str, TensorType["b ..."]]:
        """
        Build various masks for AtomMPNN.

        Ensures that the conditioning masks only contain non-pad, resolved entries.
        Also, updates batch (in place) with:
        - atomwise_token_idx: Tensor["b n_atoms", int]: index of the token that the atom belongs to, 0 for pad atoms
        - atomwise_seq_cond_mask: Tensor["b n_atoms", float]: 1 if the atom is part of an unmasked residue type, or 0 otherwise
        - token_exists_mask: Tensor["b n_tokens", float]: 1 if there exists any unmasked atom in the token, or 0 otherwise
        """            
    
        # Ensure the conditioning masks only contain non-pad, resolved entries.
        batch["seq_cond_mask"] = batch["seq_cond_mask"] * batch["token_resolved_mask"] * batch["token_pad_mask"]
        batch["atom_cond_mask"] = batch["atom_cond_mask"] * batch["atom_resolved_mask"] * batch["atom_pad_mask"]
    
        # Build mask for which tokens to include in the token-level grpah
        ## ensure center atom is present, since graph nodes are the center atom
        batch["token_exists_mask"] = batch["token_resolved_mask"].float()  # [b, n_tokens], "whether the token exists in the residue-level graph"

        ## sometimes, it's helpful to mask out certain tokens from the graph (e.g. for protein-only design in lcaliby or exclude hetero residues in sampling)
        token_exists_override = batch.get("token_exists_override", torch.ones_like(batch["token_exists_mask"]))
        batch["token_exists_mask"] = batch["token_exists_mask"] * token_exists_override
        
        # Mask out hetero residues in protein residue graphs for sampling, if specified. 
        #Todo: Need to implement functionality for redesigning hetero residues into standard AA in the future.
        residuewise_hetero_mask = batch.get("residuewise_hetero_mask", torch.ones_like(batch["token_exists_mask"]))
        atomwise_hetero_mask = batch.get("atomwise_hetero_mask", torch.ones_like(batch["atom_resolved_mask"]))
        
        if not is_sampling:
            # Exclude pseudo-context positions (pseudo-ligands + backbone-masked neighbors)
            # from the protein residue graph.  # JH Changed 260416
            pseudo_context_mask = batch.get("pseudo_context_mask", torch.zeros_like(batch["token_pad_mask"]))

            batch["protein_residue_node_mask"] = (
                batch["token_is_prot_std_aa"] *
                (1 - pseudo_context_mask) *
                batch["token_exists_mask"] *
                batch["token_pad_mask"]
            )
            
        else:
            #Todo: Need to implement functionality for redesigning hetero residues into standard AA in the future.            
            batch["protein_residue_node_mask"] = (
                batch["token_is_prot_std_aa"] *                 
                residuewise_hetero_mask *              
                batch["token_exists_mask"] *
                batch["token_pad_mask"]
            )                    
            
            batch["atom_cond_mask"] = batch["atom_cond_mask"] * atomwise_hetero_mask

        return batch

    def potts_sample(self,
                     batch: dict[str, TensorType["b ..."]],
                     sampling_inputs: dict[str, Any]
                     ) -> tuple[dict[str, list[AtomArray]], dict[str, Any]]:
        """
        Potts sampling for sequence design.

        When ``potts_sampling_cfg.guidance_cfg.enabled`` is true, a second
        forward pass is run on a ligand-masked (``protein_only``) copy of the
        batch to obtain "uncond" Potts parameters. The sampler then runs DLMC
        on the linearly-mixed parameters

            h_mix = gamma * h_cond + (1 - gamma) * h_uncond
            J_mix = gamma * J_cond + (1 - gamma) * J_uncond

        sweeping over ``gamma_list``. This samples from the Boltzmann
        distribution of ``U_mix = gamma * U_cond + (1 - gamma) * U_uncond``.
        For each sampled sequence we also record post-hoc physical Potts
        energies ``U_cond`` and ``U_uncond`` (no LCP penalty) so that
        downstream code can plot the Pareto front of ligand-fit vs.
        ligand-free stability.

        Returns:
            output_feats: list[dict[str, TensorType["b ..."]]]: list of length (n_samples_per_pdb) of output features for each sample
            aux: dict[str, Any]: auxiliary outputs
        """
        aux = {}

        # If specified, condition on sequence only in the potts model
        batch["seq_cond_mask_potts"] = batch["seq_cond_mask"].clone()
        if sampling_inputs["potts_sampling_cfg"].get("potts_only_cond", False):
            print("Conditioning on sequence only in the potts model")
            batch["seq_cond_mask"] = torch.zeros_like(batch["seq_cond_mask"])  # zero out model-level sequence conditioning mask

        # Parse guidance config (optional).
        potts_sampling_cfg = sampling_inputs["potts_sampling_cfg"]
        guidance_cfg = potts_sampling_cfg.get("guidance_cfg", None)
        use_guidance = bool(guidance_cfg) and bool(guidance_cfg.get("enabled", False))
        if use_guidance:
            if "tied_sampling_ids" in batch:
                raise NotImplementedError(
                    "Potts guidance is not supported together with tied_sampling."
                )
            gamma_list = list(guidance_cfg.get("gamma_list", [1.0]))
            schedule_list_raw = guidance_cfg.get("schedule_list", None)
            schedule_list = [dict(s) for s in schedule_list_raw] if schedule_list_raw is not None else None
            uncond_mode = guidance_cfg.get("uncond_mode", "protein_only")
            if uncond_mode != "protein_only":
                raise NotImplementedError(
                    f"Unsupported uncond_mode={uncond_mode!r}. Only 'protein_only' is implemented."
                )
            # Build the uncond batch *before* the first compute_potts_params
            # call so both branches see the same starting state. A shallow
            # copy is enough — we only rebind specific keys; no tensors are
            # mutated in place.
            batch_uncond = dict(batch)
            batch_uncond["atom_cond_mask"] = batch["atom_cond_mask"] * batch["atom_is_protein_chain"]
            pocket_distance = float(guidance_cfg.get("pocket_distance", 10.0))
        else:
            gamma_list = [None]
            schedule_list = None
            batch_uncond = None
            pocket_distance = None

        # Compute cond potts parameters
        potts_decoder_aux, batch, sampling_inputs = self.compute_potts_params(batch, sampling_inputs)

        # Compute uncond potts parameters (single extra forward pass) if guidance is on
        potts_decoder_aux_uncond = None
        pocket_mask = None
        n_protein = None
        n_pocket = None
        if use_guidance:
            potts_decoder_aux_uncond, _, _ = self.compute_potts_params(batch_uncond, sampling_inputs)
            # Pocket mask is structure-only; compute once per batch, reuse for
            # every (gamma, sample). N_pocket==0 is possible (no ligand atoms
            # or nothing within pocket_distance) — handled via clamp below.
            pocket_mask, n_protein = self._compute_ligand_pocket_mask(
                batch, pocket_distance=pocket_distance,
            )  # [B, N], [B]
            n_pocket = pocket_mask.sum(-1)  # [B]

        # Set up Potts sampling
        regularization = potts_sampling_cfg["regularization"]
        potts_sweeps = potts_sampling_cfg["potts_sweeps"]
        potts_proposal = potts_sampling_cfg["potts_proposal"]
        potts_temperature = potts_sampling_cfg["potts_temperature"]
        rejection_step = potts_sampling_cfg.get("rejection_step", potts_proposal == "chromatic")

        B, N, _ = batch["restype"].shape
        logits_init = torch.zeros((B, N, const.AF3_ENCODING.n_tokens), device=batch["restype"].device).float()

        # Handle banned amino acids and aatype restrictions
        ban_S = {"X"}
        omit_aas = sampling_inputs.get("omit_aas", None)
        if omit_aas is not None:
            ban_S = ban_S | set(omit_aas)
        ban_S = const.AF3_ENCODING.encode_aa_seq(ban_S)
        ban_S = ban_S + const.AF3_ENCODING.encode(const.AF3_ENCODING.non_protein_tokens)  # ban all non-protein tokens

        # Initialize random sequence and sampling masks
        mask_sample = (1 - batch["seq_cond_mask_potts"]) * batch["token_pad_mask"]  # 1 where we can sample, 0 where we can't

        mask_sample, _, S_init = potts.init_sampling_masks(
            logits_init, mask_sample=mask_sample, S=batch["restype"].argmax(dim=-1), ban_S=ban_S, pos_restrict_aatype=sampling_inputs.get("pos_restrict_aatype", None)
        )

        # Complexity regularization
        penalty_func = None
        mask_ij_coloring = None
        edge_idx_coloring = None
        if regularization == "LCP":
            C_complexity = batch["asym_id"] - torch.min(batch["asym_id"]) + 1  # renumber asym_id to have min value of 1
            C_complexity = C_complexity * batch["protein_residue_node_mask"]
            #! fixed, 251110
            # mask out i) non-protein chains, ii) pad tokens, iii) tokens that don't exist in the graph
            # complexity is only calculated for the residues where C_complexity > 0
            penalty_func = lambda _S: complexity.complexity_lcp(_S, C_complexity)

        S = []  # keep track of sequences for each sample
        aux["U"] = []  # mixed energy per sample (equal to U_cond when guidance is off)
        aux["gamma"] = []  # gamma used to produce each sample (None when guidance is off; γ_max for schedules)
        aux["schedule_label"] = []  # human-readable schedule tag (None when guidance is off)
        aux["U_cond"] = []  # post-hoc cond energy (no penalty); None when guidance is off
        aux["U_uncond"] = []  # post-hoc uncond energy (no penalty); None when guidance is off
        # Per-residue and pocket-restricted energy aux (guidance-only). All
        # are shape [B] per sample when filled, or None when guidance is off.
        aux["U_cond_per_res"] = []
        aux["U_uncond_per_res"] = []
        aux["U_cond_pocket"] = []
        aux["U_uncond_pocket"] = []
        aux["U_cond_pocket_per_res"] = []
        aux["U_uncond_pocket_per_res"] = []
        aux["N_pocket"] = []

        num_seqs_per_pdb = sampling_inputs["num_seqs_per_pdb"]

        # Outer iteration: schedule_list (when set) takes precedence over
        # gamma_list. Each item is normalized to {label, type, gamma_max}; for
        # constant entries we pass gamma_schedule_cfg=None so the legacy
        # constant-γ path runs unchanged.
        if use_guidance and schedule_list is not None:
            iter_items = []
            for sched in schedule_list:
                if "type" not in sched or "gamma_max" not in sched:
                    raise ValueError(
                        f"schedule_list entry must include 'type' and 'gamma_max'; got {sched!r}"
                    )
                iter_items.append({
                    "label": str(sched.get("label", f"{sched['type']}_g{float(sched['gamma_max']):.2f}")),
                    "type": str(sched["type"]),
                    "gamma_max": float(sched["gamma_max"]),
                    **({"tau": float(sched["tau"])} if "tau" in sched else {}),
                })
        else:
            iter_items = [
                {
                    "label": (f"gamma_{g:.2f}" if g is not None else "no-guidance"),
                    "type": "constant",
                    "gamma_max": (float(g) if g is not None else None),
                }
                for g in gamma_list
            ]

        # Design sequences: outer loop over schedule (or constant gamma),
        # inner loop over samples-per-pdb.
        for sched in iter_items:
            sched_label = sched["label"]
            sched_gamma_max = sched["gamma_max"]
            sched_cfg = None if sched["type"] == "constant" else sched
            desc = f"schedule={sched_label}"
            for _ in tqdm(range(num_seqs_per_pdb), desc=f"Sampling sequences ({desc})", leave=False):

                if use_guidance:
                    S_sample, U_sample = self.atom_mpnn.decoder_S_potts.sample(
                        potts_decoder_aux["h"],
                        potts_decoder_aux["J"],
                        potts_decoder_aux["edge_idx"],
                        potts_decoder_aux["mask_i"],
                        potts_decoder_aux["mask_ij"],
                        S=S_init,
                        mask_sample=mask_sample,
                        temperature=potts_temperature,
                        num_sweeps=potts_sweeps,
                        penalty_func=penalty_func,
                        proposal=potts_proposal,
                        rejection_step=rejection_step,
                        verbose=False,
                        edge_idx_coloring=edge_idx_coloring,
                        mask_ij_coloring=mask_ij_coloring,
                        h_uncond=potts_decoder_aux_uncond["h"],
                        J_uncond=potts_decoder_aux_uncond["J"],
                        edge_idx_uncond=potts_decoder_aux_uncond["edge_idx"],
                        gamma=sched_gamma_max,
                        gamma_schedule_cfg=sched_cfg,
                    )
                else:
                    S_sample, U_sample = self.atom_mpnn.decoder_S_potts.sample(
                        potts_decoder_aux["h"],
                        potts_decoder_aux["J"],
                        potts_decoder_aux["edge_idx"],
                        potts_decoder_aux["mask_i"],
                        potts_decoder_aux["mask_ij"],
                        S=S_init,
                        mask_sample=mask_sample,
                        temperature=potts_temperature,
                        num_sweeps=potts_sweeps,
                        penalty_func=penalty_func,
                        proposal=potts_proposal,
                        rejection_step=rejection_step,
                        verbose=False,
                        edge_idx_coloring=edge_idx_coloring,
                        mask_ij_coloring=mask_ij_coloring,
                    )

                # Set all tokens that don't exist in the graph to unknown
                S_sample = self._set_non_protein_tokens(S_sample, batch)

                aux["U"].append(U_sample.cpu())
                aux["gamma"].append(sched_gamma_max)
                aux["schedule_label"].append(sched_label if use_guidance else None)

                if use_guidance:
                    # Post-hoc physical Potts energies on both branches. Used
                    # as the (x, y) coordinates of the Pareto plot downstream.
                    U_cond_post, _, U_cond_per_res_post = potts.compute_potts_energy(
                        S_sample,
                        potts_decoder_aux["h"],
                        potts_decoder_aux["J"],
                        potts_decoder_aux["edge_idx"],
                        return_per_res=True,
                    )
                    U_uncond_post, _, U_uncond_per_res_post = potts.compute_potts_energy(
                        S_sample,
                        potts_decoder_aux_uncond["h"],
                        potts_decoder_aux_uncond["J"],
                        potts_decoder_aux_uncond["edge_idx"],
                        return_per_res=True,
                    )
                    # Pocket-restricted totals: sum per-residue contributions
                    # over pocket residues only.
                    U_cond_pocket = (U_cond_per_res_post * pocket_mask).sum(-1)
                    U_uncond_pocket = (U_uncond_per_res_post * pocket_mask).sum(-1)
                    safe_np = n_pocket.clamp(min=1.0)
                    safe_n = n_protein.clamp(min=1.0)
                    U_cond_per_res_global = U_cond_post / safe_n
                    U_uncond_per_res_global = U_uncond_post / safe_n
                    U_cond_pocket_per_res = U_cond_pocket / safe_np
                    U_uncond_pocket_per_res = U_uncond_pocket / safe_np

                    aux["U_cond"].append(U_cond_post.cpu())
                    aux["U_uncond"].append(U_uncond_post.cpu())
                    aux["U_cond_per_res"].append(U_cond_per_res_global.cpu())
                    aux["U_uncond_per_res"].append(U_uncond_per_res_global.cpu())
                    aux["U_cond_pocket"].append(U_cond_pocket.cpu())
                    aux["U_uncond_pocket"].append(U_uncond_pocket.cpu())
                    aux["U_cond_pocket_per_res"].append(U_cond_pocket_per_res.cpu())
                    aux["U_uncond_pocket_per_res"].append(U_uncond_pocket_per_res.cpu())
                    aux["N_pocket"].append(n_pocket.cpu())
                else:
                    aux["U_cond"].append(None)
                    aux["U_uncond"].append(None)
                    aux["U_cond_per_res"].append(None)
                    aux["U_uncond_per_res"].append(None)
                    aux["U_cond_pocket"].append(None)
                    aux["U_uncond_pocket"].append(None)
                    aux["U_cond_pocket_per_res"].append(None)
                    aux["U_uncond_pocket_per_res"].append(None)
                    aux["N_pocket"].append(None)

                S.append(S_sample.cpu())

        # Free GPU potts parameters before postprocessing
        del potts_decoder_aux
        if potts_decoder_aux_uncond is not None:
            del potts_decoder_aux_uncond

        per_sample_aux = [
            {
                "U": aux["U"][si],
                "gamma": aux["gamma"][si],
                "schedule_label": aux["schedule_label"][si],
                "U_cond": aux["U_cond"][si],
                "U_uncond": aux["U_uncond"][si],
                "U_cond_per_res": aux["U_cond_per_res"][si],
                "U_uncond_per_res": aux["U_uncond_per_res"][si],
                "U_cond_pocket": aux["U_cond_pocket"][si],
                "U_uncond_pocket": aux["U_uncond_pocket"][si],
                "U_cond_pocket_per_res": aux["U_cond_pocket_per_res"][si],
                "U_uncond_pocket_per_res": aux["U_uncond_pocket_per_res"][si],
                "N_pocket": aux["N_pocket"][si],
            }
            for si in range(len(S))
        ]
        return self._postprocess_sampled_sequences(S, batch, per_sample_aux=per_sample_aux)


    @torch.no_grad()
    def mlm_sample(self,
                   batch: dict[str, TensorType["b ..."]],
                   sampling_inputs: dict[str, Any]
                   ) -> tuple[dict[str, list[AtomArray]], dict[str, Any]]:
        """
        MLM (order-agnostic autoregressive) sampling using W_out logits.
        Iteratively unmasks positions, re-running the full model each step.

        Follows the pattern from fampnn/fampnn/model/sd_model.py:sample().
        """
        mlm_cfg = sampling_inputs["mlm_sampling_cfg"]
        num_steps = mlm_cfg["num_steps"]
        temperature = mlm_cfg["temperature"]
        num_seqs_per_pdb = sampling_inputs.get("num_seqs_per_pdb", 1)

        # Build masks once (sets protein_residue_node_mask, token_exists_mask)
        batch = self.build_masks(batch, is_sampling=True)
        B, N, C = batch["restype"].shape
        device = batch["restype"].device

        # Banned token logit bias
        ban_S = {"X"}
        omit_aas = sampling_inputs.get("omit_aas", None)
        if omit_aas is not None:
            ban_S = ban_S | set(omit_aas)
        ban_indices = const.AF3_ENCODING.encode_aa_seq(ban_S)
        ban_indices = ban_indices + const.AF3_ENCODING.encode(const.AF3_ENCODING.non_protein_tokens)
        gap_idx = const.AF3_ENCODING.token_to_idx["<G>"]
        if gap_idx not in ban_indices:
            ban_indices.append(gap_idx)

        logit_bias = torch.zeros(C, device=device)
        logit_bias[ban_indices] = -1e9

        # Timestep schedule → K values (fampnn: sd_model.py:233-235)
        timesteps = get_timesteps_from_schedule(
            mode=mlm_cfg["timestep_schedule"]["mode"],
            num_steps=mlm_cfg["timestep_schedule"]["num_steps"],
            t_start=mlm_cfg["timestep_schedule"]["t_start"],
            t_end=mlm_cfg["timestep_schedule"]["t_end"],
        ).to(device)

        # Designable positions and schedule
        original_seq_cond_mask = batch["seq_cond_mask"].clone()
        original_restype = batch["restype"].clone()
        restype_dtype = original_restype.dtype

        designable_mask = (1 - original_seq_cond_mask) * batch["protein_residue_node_mask"]
        n_designable = designable_mask.sum(dim=-1).long()           # [B]
        n_partial = original_seq_cond_mask.sum(dim=-1).long()       # [B]
        timesteps_K = torch.ceil(
            timesteps[None, :] * n_designable[:, None].float()
        ).long() + n_partial[:, None]                                # [B, S+1]

        gap_onehot = F.one_hot(
            torch.tensor(gap_idx, device=device), num_classes=C
        ).to(restype_dtype)

        # Main sampling loop
        S_all = []
        for sample_idx in tqdm(range(num_seqs_per_pdb), desc="MLM sampling sequences", leave=False):
            # Reset state: mask designable positions to gap token
            seq_cond_mask = original_seq_cond_mask.clone()
            restype = original_restype.clone()
            restype[designable_mask.bool()] = gap_onehot

            # Random decoding order (fampnn: sampling_utils.get_decoding_order)
            decoding_order = get_decoding_order(
                mode=mlm_cfg.get("aatype_decoding_order_mode", "random"),
                seq_mask=designable_mask,
                mlm_mask_prev=seq_cond_mask,
            )

            # Iterative unmasking (fampnn: sd_model.py:237-271)
            for step in range(num_steps):
                K_next = timesteps_K[:, step + 1]  # [B]

                batch["seq_cond_mask"] = seq_cond_mask
                batch["restype"] = restype

                # Forward pass → logits
                seq_logits, _ = self.atom_mpnn(batch, is_sampling=True)

                # Sample tokens (fampnn: fampnn_denoiser.py:112-130)
                masked_logits = seq_logits + logit_bias[None, None, :]
                if temperature == 0.0:
                    aatype_pred = masked_logits.argmax(dim=-1)
                else:
                    probs = F.softmax(masked_logits / temperature, dim=-1)
                    aatype_pred = torch.multinomial(probs.view(-1, C), 1).view(B, N)

                # Update mask (fampnn: sampling_utils.update_mlm_mask)
                seq_mlm_mask_prev = seq_cond_mask.clone()
                newly_unmask = (~seq_cond_mask.bool()) & (decoding_order < K_next[:, None])
                seq_cond_mask = (newly_unmask.float() + seq_cond_mask).clamp(max=1.0)

                # Update restype at newly unmasked positions (fampnn: sampling_utils.unmask)
                aatype_pred_onehot = F.one_hot(aatype_pred, num_classes=C).to(restype_dtype)
                restype = torch.where(newly_unmask.unsqueeze(-1), aatype_pred_onehot, restype)

            # Collect final sequence
            S_sample = restype.argmax(dim=-1)  # [B, N]
            S_sample = self._set_non_protein_tokens(S_sample, batch)
            S_all.append(S_sample.cpu())

        # Restore original batch state
        batch["seq_cond_mask"] = original_seq_cond_mask
        batch["restype"] = original_restype

        return self._postprocess_sampled_sequences(S_all, batch)


    @staticmethod
    def _compute_ligand_pocket_mask(
        batch: dict[str, torch.Tensor],
        pocket_distance: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-batch ligand pocket mask over token positions.

        A protein residue is "in the pocket" iff its pseudo-Cβ coordinate is
        within ``pocket_distance`` Å of any resolved ligand atom in the same
        batch item. The pseudo-Cβ is read from
        ``batch['noised_pseudo_cb_coords']``, which at sequence-design time
        has zero structure noise and so matches the real pseudo-Cβ.

        Returns:
            pocket_mask: float tensor ``[B, N_tokens]``, 1 at pocket residues,
                0 elsewhere.
            n_protein: float tensor ``[B]``, count of valid protein residues
                per batch item (used for whole-protein per-residue averaging).
        """
        pcb = batch["noised_pseudo_cb_coords"]                # [B, N, 3]
        coords = batch["coords"]                              # [B, N_atoms, 3]
        lig_atom_mask = (
            batch["atom_is_small_molecule_chain"].bool()
            & batch["atom_resolved_mask"].bool()
            & batch["atom_pad_mask"].bool()
        )                                                     # [B, N_atoms]

        # Entries with zero pseudo-Cβ flag non-standard / unresolved tokens.
        # Combine with the existing protein-residue node mask to be safe.
        pcb_valid = (
            (pcb.norm(dim=-1) > 1e-6)
            & batch["protein_residue_node_mask"].bool()
        )                                                     # [B, N]
        n_protein = pcb_valid.sum(-1).float()                 # [B]

        B, N, _ = pcb.shape
        pocket = torch.zeros((B, N), device=pcb.device, dtype=torch.float32)
        d2_threshold = float(pocket_distance) ** 2
        for b in range(B):
            if not lig_atom_mask[b].any():
                continue
            lig_b = coords[b][lig_atom_mask[b]]               # [L_b, 3]
            pcb_b = pcb[b]                                    # [N, 3]
            d2 = ((pcb_b[:, None, :] - lig_b[None, :, :]) ** 2).sum(-1)
            min_d2 = d2.min(dim=1).values                     # [N]
            pocket[b] = (min_d2 < d2_threshold).float()
        pocket = pocket * pcb_valid.float()
        return pocket, n_protein

    @staticmethod
    def _set_non_protein_tokens(S: TensorType["b n", int],
                                batch: dict[str, TensorType["b ..."]],
                                ) -> TensorType["b n", int]:
        """Set non-protein-residue-node positions to appropriate unknown tokens."""
        non_protein = ~batch["protein_residue_node_mask"].bool()
        S = torch.where(non_protein & (batch["is_protein"] | batch["is_ligand"]),
                        const.AF3_ENCODING.token_to_idx[const.UNKNOWN_AA], S)
        S = torch.where(non_protein & batch["is_rna"],
                        const.AF3_ENCODING.token_to_idx[const.UNKNOWN_RNA], S)
        S = torch.where(non_protein & batch["is_dna"],
                        const.AF3_ENCODING.token_to_idx[const.UNKNOWN_DNA], S)
        return S


    def _postprocess_sampled_sequences(
        self,
        S_list: list[TensorType["b n", int]],
        batch: dict[str, TensorType["b ..."]],
        per_sample_aux: list[dict] | None = None,
    ) -> tuple[dict[str, list[AtomArray]], dict[str, Any]]:
        """Thread sampled integer sequences onto atom arrays.

        Args:
            S_list: list of [B, N] integer tensors, one per sample.
            batch: the batch dict (will be moved to CPU).
            per_sample_aux: optional list of dicts with per-sample auxiliary data (e.g. {"U": tensor}).
        """
        batch = to(batch, device="cpu")

        id_to_atom_arrays = defaultdict(list)
        id_to_aux = defaultdict(list)
        for si in range(len(S_list)):
            atom_arrays = copy.deepcopy(batch["atom_array"])

            for bi in range(len(atom_arrays)):
                token_pad_mask = batch["token_pad_mask"][bi].bool()
                atom_pad_mask = batch["atom_pad_mask"][bi].bool()

                new_restype = S_list[si][bi][token_pad_mask]
                new_coords = batch["coords"][bi][atom_pad_mask]

                example_id = batch["example_id"][bi]
                atom_array = atom_arrays[bi]
                seq_cond_mask = batch["seq_cond_mask"][bi][token_pad_mask]
                atom_cond_mask = batch["atom_cond_mask"][bi][atom_pad_mask]
                atom_resolved_mask = batch["atom_resolved_mask"][bi][atom_pad_mask]

                # Update resnames.
                update_seq_mask = ~seq_cond_mask.numpy().astype(bool)
                atomwise_update_seq_mask = spread_token_wise(atom_array, update_seq_mask)
                atomwise_resnames = spread_token_wise(atom_array, const.AF3_ENCODING.idx_to_token[new_restype])
                atomwise_resnames = np.where(atomwise_update_seq_mask,
                                             atomwise_resnames,
                                             atom_array.get_annotation("res_name"))
                atom_array.set_annotation("res_name", atomwise_resnames)

                # Update coords.
                update_coords_mask = (atom_cond_mask * atom_resolved_mask).numpy().astype(bool)
                atom_array.coord = np.where(update_coords_mask[..., None],
                                            new_coords.numpy(),
                                            np.nan)

                id_to_atom_arrays[example_id].append(atom_array)

                # Auxiliary outputs.
                sample_aux = {"S": new_restype.cpu()}

                def _extract_scalar(entry, batch_idx):
                    if entry is None:
                        return float("nan")
                    if torch.is_tensor(entry):
                        return entry[batch_idx].cpu().item()
                    return float("nan")

                if per_sample_aux is not None:
                    aux_si = per_sample_aux[si]
                    sample_aux["U"] = _extract_scalar(aux_si.get("U"), bi)
                    if "gamma" in aux_si:
                        sample_aux["gamma"] = aux_si["gamma"]  # scalar or None
                    if "schedule_label" in aux_si:
                        sample_aux["schedule_label"] = aux_si["schedule_label"]  # str or None
                    for key in (
                        "U_cond",
                        "U_uncond",
                        "U_cond_per_res",
                        "U_uncond_per_res",
                        "U_cond_pocket",
                        "U_uncond_pocket",
                        "U_cond_pocket_per_res",
                        "U_uncond_pocket_per_res",
                        "N_pocket",
                    ):
                        if key in aux_si:
                            sample_aux[key] = _extract_scalar(aux_si.get(key), bi)
                else:
                    sample_aux["U"] = float("nan")
                id_to_aux[example_id].append(sample_aux)

        return id_to_atom_arrays, id_to_aux


    def compute_potts_params(self, batch: dict[str, TensorType["b ..."]],
                             sampling_inputs: dict[str, Any]) -> tuple[dict[str, TensorType["b ..."]], dict[str, TensorType["b ..."]], dict[str, Any]]:
        """
        Run model and collect potts parameters over a batch of samples.

        If "tied_sampling_ids" is in batch, we will aggregate potts parameters across tied groups and slice batch to representative elements.

        Returns:
            potts_decoder_aux: dict[str, TensorType["b ..."]]: potts parameters
            batch: dict[str, TensorType["b ..."]]: batch with token_exists_mask added
            sampling_inputs: dict[str, Any]: sampling inputs with pos_restrict_aatype sliced to representative elements
        """
        subbatch_size = sampling_inputs["batch_size"]
        B = batch["restype"].shape[0]

        # Run model and collect potts parameters
        potts_decoder_aux = {}  # potts parameters
        token_exists_mask = []
        protein_residue_node_mask = []  # keep track of the residues that exist in the graph
        for bi in tqdm(range(0, B, subbatch_size), desc="Computing potts parameters", leave=False):
            subbatch = slice_feats(batch, slice(bi, bi + subbatch_size))

            _, aux_preds_i = self(subbatch, is_sampling=True, sampling_inputs=sampling_inputs)

            for k, v in aux_preds_i["potts_decoder_aux"].items():
                potts_decoder_aux.setdefault(k, []).append(v)
            protein_residue_node_mask.append(aux_preds_i["protein_residue_node_mask"])
            token_exists_mask.append(aux_preds_i["token_exists_mask"])
            del aux_preds_i  # free seq_logits, h_V, h_ESV etc.
        potts_decoder_aux = {k: torch.cat(v, dim=0) for k, v in potts_decoder_aux.items()}
        
        token_exists_mask = torch.cat(token_exists_mask, dim=0)
        protein_residue_node_mask = torch.cat(protein_residue_node_mask, dim=0)
        batch["protein_residue_node_mask"] = protein_residue_node_mask  # store in batch for downstream use
        batch["token_exists_mask"] = token_exists_mask  # store in batch for downstream use
        
        # Handle tied sampling
        if "tied_sampling_ids" in batch:
            tied_sampling_inputs = _construct_tied_sampling_inputs(batch)

            # slice to representative elements
            unique_rep_idxs = tied_sampling_inputs["rep_idx"].unique().tolist()
            batch = slice_feats(batch, unique_rep_idxs)  # get representative batch elements

            if sampling_inputs.get("pos_restrict_aatype", None) is not None:
                sampling_inputs["pos_restrict_aatype"] = [x[unique_rep_idxs] for x in sampling_inputs["pos_restrict_aatype"]]

            # aggregate potts parameters across tied groups
            potts_decoder_aux = _aggregate_potts_params(potts_decoder_aux, tied_sampling_inputs)

        return potts_decoder_aux, batch, sampling_inputs


def _aggregate_potts_params(potts_decoder_aux: dict[str, TensorType["b ..."]],
                            tied_sampling_inputs: dict[str, Any],
                            use_mean: bool = True,
                            ) -> dict[str, TensorType["b ..."]]:
    """
    Aggregate potts parameters across tied groups.

    If use_mean, we take the mean of the potts parameters across the tied groups (equivalent to geometric mean in probability space)
    """
    h, J, edge_idx, mask_i, mask_ij = potts_decoder_aux["h"], potts_decoder_aux["J"], potts_decoder_aux["edge_idx"], potts_decoder_aux["mask_i"], potts_decoder_aux["mask_ij"]
    inverse, unique_ids = tied_sampling_inputs["inverse"], tied_sampling_inputs["unique_ids"]

    # handle 1D features
    counts = torch.bincount(inverse)
    h_new = h.new_zeros(unique_ids.shape[0], *h.shape[1:]).index_add(0, inverse, h)
    node_counts = mask_i.new_zeros(unique_ids.shape[0], *mask_i.shape[1:]).index_add(0, inverse, mask_i)
    mask_i_new = (node_counts == counts.view(-1, 1)).float()  # node i is unmasked only if node i is present across all inputs in the tied group

    # handle 2D features
    n_grp = unique_ids.shape[0]
    B, N, K = edge_idx.shape
    C = J.shape[-1]
    edge_counts = mask_ij.new_zeros(n_grp, N, N)
    J_new = J.new_zeros(n_grp, N, N, C, C)
    for bi in range(B):
        g = inverse[bi]

        edge_indices_flat = (edge_idx[bi] + torch.arange(N, device=edge_idx.device)[:, None] * N).reshape(-1)
        edge_counts[g].view(-1).index_add_(0, edge_indices_flat, mask_ij[bi].view(-1))  # count number of edges between each pair of nodes
        J_new[g].view(-1, C, C).index_add_(0, edge_indices_flat, J[bi].view(-1, C, C))  # add in the pairwise interactions for this graph

    mask_ij_new = (edge_counts > 0) * (mask_i_new[:, :, None] * mask_i_new[:, None, :])  # edge i,j is present only if both nodes are present and there exists some edge between them
    edge_idx_new = torch.arange(N, device=edge_idx.device).expand(1, 1, -1).repeat(n_grp, N, 1)  # new edge indices are given in the full NxN grid

    if use_mean:
        J_new = J_new / counts.view(-1, 1, 1, 1, 1)
        h_new = h_new / counts.view(-1, 1, 1)

    potts_decoder_aux_new = {
        "h": h_new,
        "J": J_new,
        "edge_idx": edge_idx_new,
        "mask_i": mask_i_new,
        "mask_ij": mask_ij_new,
    }

    return potts_decoder_aux_new


def _construct_tied_sampling_inputs(batch: dict[str, TensorType["b ..."]]) -> dict[str, Any]:
    tied_sampling_inputs = {"tied_sampling_ids": batch["tied_sampling_ids"]}
    device = batch["tied_sampling_ids"].device
    tied_sampling_inputs["unique_ids"], tied_sampling_inputs["inverse"] = tied_sampling_inputs["tied_sampling_ids"].unique(return_inverse=True)

    # use first index of each tied group as the representative index
    B = batch["res_type"].shape[0]
    batch_idx = torch.arange(B, device=device)
    n_unique_ids = tied_sampling_inputs["unique_ids"].shape[0]
    first_idxs = torch.full((n_unique_ids, ), B, device=device)
    first_idxs.scatter_reduce_(0, tied_sampling_inputs["inverse"], batch_idx, reduce="amin", include_self=True)
    tied_sampling_inputs["rep_idx"] = first_idxs[tied_sampling_inputs["inverse"]]
    return tied_sampling_inputs
