from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType

import allatom_design.data.residue_constants as rc
from allatom_design.data.data import cat_bb_scn, get_rc_tensor
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

        # Sequence design model: AtomMPNN
        self.atom_mpnn = AtomMPNN(cfg.mpnn)


    def forward(self,
                batch: dict[str, TensorType["b ..."]],
                ) -> Tuple[TensorType["b n a 3", float],  # x1 pred
                           TensorType["b n", int],  # aatype pred
                           dict[str, TensorType["b ..."]]  # aux_preds
                           ]:
        mpnn_features = self.atom_mpnn(batch)



        # # Construct atom_mask_noised: 0 for missing / ghost / masked / pad atoms, 1 otherwise
        # atom_mask_noised = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype_noised)  # 0 for ghost atoms; X only has backbone atoms
        # atom_mask_noised = atom_mask_noised * seq_mask.unsqueeze(-1)  # mask out padding
        # atom_mask_noised = atom_mask_noised * (1 - missing_atom_mask)  # mask out missing atoms
        # atom_mask_noised[..., rc.non_bb_idxs] = atom_mask_noised[..., rc.non_bb_idxs] * scn_mlm_mask.unsqueeze(-1)  # mask out masked sidechain atoms

        # # Sequence design
        # seq_logits, mpnn_feature_dict = self.seq_design_module(
        #     x_noised,
        #     aatype_noised,
        #     seq_mask,
        #     atom_mask_noised,
        #     residue_index,
        #     chain_encoding,
        #     noise=aux_inputs.get("noise", None),
        #     noise_labels=aux_inputs.get("noise_labels", None),
        #     h_S_init=esmc_embed)

        # aatype_pred, scaled_seq_probs = self.sample_aatype(seq_logits, aux_inputs, is_sampling)

        # # Outputs
        # aux_preds = {
        #     "seq_logits": seq_logits,
        #     "seq_probs": F.softmax(seq_logits, dim=-1),
        #     "scaled_seq_probs": scaled_seq_probs,
        #     "potts_decoder_aux": mpnn_feature_dict.get("potts_decoder_aux", None),
        # }


        return x1_pred, aatype_pred, aux_preds


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
