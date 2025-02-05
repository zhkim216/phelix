from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchtyping import TensorType

import allatom_design.data.residue_constants as rc
from allatom_design.data.data import cat_bb_scn, get_rc_tensor
from allatom_design.model.seq_denoiser.denoisers.denoiser import \
    BaseSeqDenoiser
from allatom_design.model.seq_denoiser.denoisers.seq_design.fampnn import \
    FaMPNN
from allatom_design.model.seq_denoiser.denoisers.sidechain_diffusion.scn_diffusion_mlp import \
    SidechainDiffusionModule


class MiniMPNNDenoiser(BaseSeqDenoiser):
    def __init__(self,
                 cfg: DictConfig,
                 sigma_data: Tuple[TensorType[(), float], TensorType[(), float]]):
        super().__init__()

        self.cfg = cfg
        self.bb_sigma_data, self.scn_sigma_data = sigma_data
        self.task = cfg.task
        self.use_scn_diffusion = self.task in ["allatom_seq_des", 'scn_pack']

        # Sequence design model: MiniMPNN
        self.seq_design_module = FaMPNN(cfg.minimpnn)

        # Sidechain diffusion head: DiT
        if self.use_scn_diffusion:
            self.scn_diffusion_module = SidechainDiffusionModule(cfg.scn_diffusion_module, self.scn_sigma_data)


    def forward(self,
                x_noised: TensorType["b n a 3", float],
                aatype_noised: TensorType["b n", int],
                t: TensorType["b", float],  # possibly a tuple (t_seq, t_scn)
                residue_index: TensorType["b n", int],
                chain_encoding: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                missing_atom_mask: TensorType["b n a", float],  # 1 denotes missing atoms
                scn_mlm_mask: TensorType["b n", float],  # denotes masked sidechains
                cond_labels_in: Dict[str, TensorType["b", int]] = {},
                aux_inputs: Optional[Dict] = None,  # stores additional inputs for the model (different for training and sampling)
                is_sampling: bool = False,
                ) -> Tuple[TensorType["b n a 3", float],  # x1 pred
                           TensorType["b n", int],  # aatype pred
                           Dict[str, TensorType["b ..."]]  # aux_preds
                           ]:
        # Construct atom_mask_noised: 0 for missing / ghost / masked / pad atoms, 1 otherwise
        atom_mask_noised = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype_noised)  # 0 for ghost atoms; X only has backbone atoms
        atom_mask_noised = atom_mask_noised * seq_mask.unsqueeze(-1)  # mask out padding
        atom_mask_noised = atom_mask_noised * (1 - missing_atom_mask)  # mask out missing atoms
        atom_mask_noised[..., rc.non_bb_idxs] = atom_mask_noised[..., rc.non_bb_idxs] * scn_mlm_mask.unsqueeze(-1)  # mask out masked sidechain atoms

        # 1. Sequence design
        seq_logits, mpnn_feature_dict = self.seq_design_module(
            x_noised,
            aatype_noised,
            seq_mask,
            atom_mask_noised,
            residue_index,
            chain_encoding,
            noise_labels=aux_inputs.get("noise_labels", None))

        aatype_pred, scaled_seq_probs = self.sample_aatype(seq_logits, aux_inputs, is_sampling)

        # Outputs
        aux_preds = {
            "seq_logits": seq_logits,
            "seq_probs": F.softmax(seq_logits, dim=-1),
            "scaled_seq_probs": scaled_seq_probs,
            'seq_mask': seq_mask,
            'seq_mlm_mask': aux_inputs.get("seq_mlm_mask", None),  # used during training
            'scn_mlm_mask': aux_inputs.get('scn_mlm_mask', None)  # used during training
        }

        # 2. Sidechain diffusion
        x1_pred = None
        if self.use_scn_diffusion:
            x1_scn_pred, scn_diffusion_aux = self.scn_diffusion_module.sidechain_diffusion(
                mpnn_feature_dict,
                aatype_pred,
                seq_mask=seq_mask,
                residue_index=residue_index,
                chain_index=chain_encoding,
                aux_inputs=aux_inputs,
                is_sampling=is_sampling
            )

            aux_preds['scn_diffusion_aux'] = scn_diffusion_aux

            if is_sampling:
                # store the predicted sidechain coordinates with known backbone
                x_bb = mpnn_feature_dict["X"][..., rc.atom14_bb_idxs, :]
                x1_pred = cat_bb_scn(x_bb, x1_scn_pred)


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
        omit_aas = aux_inputs.get("omit_aas", [])
        for aa in omit_aas:
            seq_logits[..., rc.restype_order_with_x[aa]] = -1e9  # omit the specified aatypes

        restrict_pos_aatype = aux_inputs.get("restrict_pos_aatype", None)
        if restrict_pos_aatype is not None:
            restrict_pos_mask, allowed_aatype_mask = restrict_pos_aatype  # (B, N), (B, N, K)
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


    def get_sidechain_likelihoods(self,
                                  num_steps: int,
                                  x: TensorType["b n a 3", float],
                                  aatype: TensorType["b n", int],
                                  residue_index: TensorType["b n", int],
                                  chain_index: TensorType["b n", int],
                                  seq_mask: TensorType["b n", float],
                                  cond_labels_in: Dict[str, TensorType["b", int]] = {},
                                  aux_inputs: Optional[Dict] = None,  # stores additional inputs for the model (different for training and sampling)
                                  ):
        # 1. MiniMPNN on ground truth aatype for sequence embeddings
        _, h_V, _, _ = self.seq_design_module(x, aatype, seq_mask, residue_index, chain_index)

        # 2. Get sidechain likelihoods
        x1_scn, x_bb = x[..., rc.non_bb_idxs, :], x[..., rc.bb_idxs, :]
        likelihood_aux = self.scn_diffusion_module.get_likelihoods(num_steps, x1_scn, h_V, aatype, x_bb, seq_mask, residue_index, chain_index, aux_inputs)

        return likelihood_aux