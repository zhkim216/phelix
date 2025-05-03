from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType

import allatom_design.data.const as const
import allatom_design.model.seq_denoiser.denoisers.seq_design.potts as potts
from allatom_design.model.seq_denoiser.denoisers.denoiser import \
    BaseSeqDenoiser
from allatom_design.model.seq_denoiser.denoisers.seq_design.atom_mpnn import \
    AtomMPNN
from chroma.layers import complexity


class AtomMPNNDenoiser(BaseSeqDenoiser):
    def __init__(self,
                 cfg: DictConfig,
                 sigma_data: Tuple[TensorType[(), float], TensorType[(), float]]):
        super().__init__()

        self.cfg = cfg
        self.bb_sigma_data, self.scn_sigma_data = sigma_data
        self.task = cfg.task

        # Random Gaussian noise
        self.augment_eps = cfg.augment_eps
        self.per_residue_eps = cfg.per_residue_eps
        self.max_eps = cfg.max_eps

        # Sequence design model: AtomMPNN
        self.atom_mpnn = AtomMPNN(cfg.mpnn)


    def forward(self,
                batch: dict[str, TensorType["b ..."]],
                is_sampling: bool = False,
                sampling_inputs: dict[str, Any] | None = None,
                ) -> Tuple[TensorType["b n c", float],  # seq_logits
                           dict[str, TensorType["b ..."]]]:
        # Build some helpful masks based on conditioning sequence and atoms
        batch = self.build_masks(batch)

        # During training, add random noise to input coordinates
        if not is_sampling:
            batch = self.get_training_random_noise(batch)

        # Run model
        seq_logits, mpnn_feats = self.atom_mpnn(batch)

        # Outputs
        aux_preds = {
            "seq_logits": seq_logits,
            "potts_decoder_aux": mpnn_feats.get("potts_decoder_aux", None),
            "seq_cond_mask": batch["seq_cond_mask"],
            "atom_cond_mask": batch["atom_cond_mask"],
            "token_exists_mask": batch["token_exists_mask"],
        }

        return seq_logits, aux_preds


    def potts_sample(self, batch: dict[str, TensorType["b ..."]], sampling_inputs: dict[str, Any]):
        potts_sampling_cfg = sampling_inputs["potts_sampling_cfg"]
        regularization = potts_sampling_cfg["regularization"]
        potts_sweeps = potts_sampling_cfg["potts_sweeps"]
        potts_proposal = potts_sampling_cfg["potts_proposal"]
        potts_temperature = potts_sampling_cfg["potts_temperature"]

        B, N, _ = batch["res_type"].shape
        logits_init = torch.zeros((B, N, len(const.tokens)), device=batch["res_type"].device).float()

        # Handle banned amino acids
        ban_S = {"X"}
        omit_aas = sampling_inputs.get("omit_aas", None)
        if omit_aas is not None:
            ban_S = ban_S | set(omit_aas)
        ban_S = [const.prot_only_token_to_id[const.prot_letter_to_token[aa]] for aa in ban_S]

        # Initialize random sequence and sampling masks
        # first, convert res_type to protein token vocabulary
        target_res_type = batch["res_type"].argmax(dim=-1)  # undo one-hot encoding
        lookup = const.prot_only_tokens_to_all_tokens.T.to(target_res_type.device)  # [33, 21]
        target_res_type = lookup[target_res_type].argmax(dim=-1)  # [b, n]

        mask_sample, _, S_init = potts.init_sampling_masks(
            logits_init, mask_sample=(1 - batch["seq_cond_mask"]), S=target_res_type, ban_S=ban_S
        )

        # Complexity regularization
        penalty_func = None
        mask_ij_coloring = None
        edge_idx_coloring = None
        symmetry_order = None
        if regularization == "LCP":
            # C_complexity = (
            #     C
            #     if symmetry_order is None
            #     else C[:, : C.shape[1] // symmetry_order]
            # )
            C_complexity = batch["token_pad_mask"].clone()  # TODO: is C for multi-chain?
            penalty_func = lambda _S: complexity.complexity_lcp(_S, C_complexity)
            # edge_idx_coloring, mask_ij_coloring = complexity.graph_lcp(C, edge_idx, mask_ij)

        _, aux_preds = self(batch, is_sampling=True, sampling_inputs=sampling_inputs)
        potts_decoder_aux = aux_preds["potts_decoder_aux"]
        S_sample, _ = self.atom_mpnn.decoder_S_potts.sample(
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
            rejection_step=(potts_proposal == "chromatic"),
            verbose=False,
            edge_idx_coloring=edge_idx_coloring,
            mask_ij_coloring=mask_ij_coloring,
        )
        return S_sample



    def build_masks(self, batch: dict[str, TensorType["b ..."]]) -> dict[str, TensorType["b ..."]]:
        """
        Build various masks for AtomMPNN.

        Updates batch (in place) with:
        - atomwise_seq_cond_mask: Tensor["b n_atoms", float]: 1 if the atom is part of an unmasked residue type, or 0 otherwise
        - token_exists_mask: Tensor["b n_tokens", float]: 1 if there exists any unmasked atom in the token, or 0 otherwise
        """
        # Create atom-level mask which is 1 if the atom is part of an unmasked residue type, or 0 otherwise
        B, N_atoms, N_tokens = batch["atom_to_token"].shape
        batch["atomwise_seq_cond_mask"] = torch.bmm(batch["atom_to_token"].float(), batch["seq_cond_mask"].unsqueeze(-1)).squeeze(dim=-1)  # [b, n_atoms]

        # Create token-level mask which is 1 if there exists any unmasked atom in the token, or 0 otherwise

        # if using average of all atoms in token for graph nodes
        # token_n_cond_atoms = torch.bmm(batch["atom_to_token"].float().transpose(1, 2), batch["atom_cond_mask"].unsqueeze(-1)).squeeze(dim=-1)  # [b, n_tokens]
        # batch["token_exists_mask"] = (token_n_cond_atoms > 0).float()  # [b, n_tokens], "whether the token exists in the residue-level graph"

        # if using center atom of token for graph nodes: ensure center atom is present
        batch["token_exists_mask"] = batch["token_resolved_mask"].float()  # [b, n_tokens], "whether the token exists in the residue-level graph"
        return batch


    def get_training_random_noise(self, batch: dict[str, TensorType["b ..."]]) -> dict[str, TensorType["b ..."]]:
        """
        During training, adds random noise and noise labels for input coordinates.

        Updates batch (in place) with:
        - noise: Tensor["b n_atoms 3", float]: random noise for each atom
        - noise_labels: Tensor["b n_tokens", float]: noise label for each token
        """
        if not self.training or self.augment_eps <= 0:
            # if not training or no noise, and not provided, then we assume no noise
            batch["noise"] = batch.get("noise", None)
            batch["noise_labels"] = batch.get("noise_labels", None)
            return batch

        ## Training: choose random backbone noise ##
        B, N_atoms, N_tokens = batch["atom_to_token"].shape
        device = batch["atom_to_token"].device

        if self.per_residue_eps:
            # per-residue noise. Unlike Cho et al., we sample noise stds from a uniform distribution and apply different noise to each atom in a residue
            # randomly sample noise labels
            noise_labels = torch.rand((B, N_tokens), device=device) * self.augment_eps  # sample std for each residue from uniform [0, augment_eps]
            atomwise_noise_labels = torch.bmm(batch["atom_to_token"].float(), noise_labels.unsqueeze(-1)).squeeze(dim=-1)  # [b, n_atoms]
            noise = torch.randn((B, N_atoms, 3), device=device) * atomwise_noise_labels.unsqueeze(-1)
            noise = noise * batch["atom_cond_mask"].unsqueeze(-1)
        else:
            # global noise, similar to ProteinMPNN
            # add randomly sampled noise to input
            noise = self.augment_eps * torch.randn((B, N_atoms, 3), device=device)
            noise_labels = None

        batch["noise"] = noise
        batch["noise_labels"] = noise_labels
        return batch


    def sample_aatype(self,
                      seq_logits: TensorType["b n k", float],
                      aux_inputs: Dict[str, Any],
                      is_sampling: bool,
                      ) -> Tuple[TensorType["b n", int], TensorType["b n k", float]]:
        """
        Sample aatype from seq logits
        If training, just take argmax (this will be teacher-forced to the ground truth aatype during sidechain diffusion)
        If sampling, sample from (possibly temperature-scaled) logits

        Returns:
        - aatype_pred: Tensor["b n", int]
        - scaled_seq_probs: Tensor["b n k", float]: seq_probs scaled by temperature and sampling modifications
        """
        if not is_sampling:
            return seq_logits.argmax(dim=-1), F.softmax(seq_logits, dim=-1)

        # Handle aatype restrictions
        seq_logits[..., rc.restype_order_with_x["X"]] = -1e9  # do not sample mask/unknowns
        omit_aas = aux_inputs.get("omit_aas", None)
        if omit_aas is not None:
            for aa in omit_aas:
                seq_logits[..., rc.restype_order_with_x[aa]] = -1e9  # omit the specified aatypes

        pos_restrict_aatype = aux_inputs.get("pos_restrict_aatype", None)
        if pos_restrict_aatype is not None:
            restrict_pos_mask, allowed_aatype_mask = pos_restrict_aatype  # (B, N), (B, N, K)
            restrict_pos_mask = restrict_pos_mask.unsqueeze(-1).expand_as(seq_logits)
            disallowed_positions = (restrict_pos_mask == 1.0) & (allowed_aatype_mask == 0.0)  # only allow specified aatypes
            seq_logits[disallowed_positions] = -1e9

        # Handle temperature scaling
        tau = aux_inputs.get("temperature", 1.0)
        B, N = seq_logits.shape[:2]
        if tau == 0.0:
            aatype_pred = seq_logits.argmax(dim=-1)
            scaled_seq_probs = F.softmax(seq_logits, dim=-1)  # don't scale for argmax sampling
        else:
            scaled_logits = seq_logits / tau
            scaled_seq_probs = F.softmax(scaled_logits, dim=-1)
            aatype_pred = torch.multinomial(scaled_seq_probs.view(B * N, -1), num_samples=1).view(B, N)
        return aatype_pred, scaled_seq_probs
