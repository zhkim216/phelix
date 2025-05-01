from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType

import allatom_design.data.residue_constants as rc
from allatom_design.model.seq_denoiser.denoisers.denoiser import \
    BaseSeqDenoiser
from allatom_design.model.seq_denoiser.denoisers.seq_design.atom_mpnn import \
    AtomMPNN


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
                ) -> Tuple[TensorType["b n a 3", float],  # x1 pred
                           TensorType["b n", int],  # aatype pred
                           dict[str, TensorType["b ..."]]  # aux_preds
                           ]:
        # Build some helpful masks based on conditioning sequence and atoms
        batch = self.build_masks(batch)

        # During training, add random noise to input coordinates
        batch = self.get_random_noise(batch)

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
        # token_n_cond_atoms = torch.bmm(batch["atom_to_token"].float().transpose(1, 2), batch["atom_cond_mask"].unsqueeze(-1)).squeeze(dim=-1)  # [b, n_tokens]
        # batch["token_exists_mask"] = (token_n_cond_atoms > 0).float()  # [b, n_tokens], "whether the token exists in the residue-level graph"
        batch["token_exists_mask"] = batch["token_resolved_mask"].float()
        return batch



    def get_random_noise(self, batch: dict[str, TensorType["b ..."]]) -> dict[str, TensorType["b ..."]]:
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
