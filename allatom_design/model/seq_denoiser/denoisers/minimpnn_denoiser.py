from typing import Dict, Optional, Tuple

import torch.nn.functional as F
from omegaconf import DictConfig
from torchtyping import TensorType

import allatom_design.data.residue_constants as rc
from allatom_design.data.data import cat_bb_scn
from allatom_design.model.seq_denoiser.denoisers.denoiser import \
    BaseSeqDenoiser
from allatom_design.model.seq_denoiser.denoisers.sidechain_diffusion.scn_diffusion_dit import \
    SidechainDiffusionModule
from allatom_design.model.seq_denoiser.denoisers.seq_design.minimpnn import \
    MiniMPNN
from openfold.model.primitives import Linear


class MiniMPNNDenoiser(BaseSeqDenoiser):
    def __init__(self,
                 cfg: DictConfig,
                 sigma_data: Tuple[TensorType[(), float], TensorType[(), float]]):
        super().__init__()

        self.cfg = cfg
        self.bb_sigma_data, self.scn_sigma_data = sigma_data
        self.use_scn_diffusion = cfg.task in ["allatom_seq_des"]

        # Sequence design model: MiniMPNN
        self.seq_design_module = MiniMPNN(cfg.minimpnn)
        self.seq_head = Linear(cfg.seq_head.in_channels, cfg.seq_head.n_aatype, init="final")

        # Sidechain diffusion head: DiT
        if self.use_scn_diffusion:
            self.proj_z = Linear(cfg.minimpnn.n_channel, cfg.scn_diffusion_module.hidden_size)  # project h_V to conditioning input
            self.scn_diffusion_module = SidechainDiffusionModule(cfg.scn_diffusion_module, self.scn_sigma_data)


    def forward(self,
                x_noised: TensorType["b n a 3", float],
                aatype_noised: Optional[TensorType["b n", int]],
                t: TensorType["b", float],  # possibly a tuple (t_seq, t_scn)
                residue_index: TensorType["b n", int],
                seq_mask: TensorType["b n", float],
                seq_self_cond: Optional[TensorType["b n k", float]] = None,  # k = n_aatype, logits
                cond_labels_in: Dict[str, TensorType["b", int]] = {},
                aux_inputs: Optional[Dict] = None,  # stores additional inputs for the model (different for training and sampling)
                is_sampling: bool = False,
                ) -> Tuple[TensorType["b n a 3", float],  # x1 pred
                           TensorType["b n", int],  # aatype pred
                           Dict[str, TensorType["b ..."]]  # aux_preds
                           ]:
        aux_preds = {}

        # 1. MiniMPNN for sequence design
        _, h_V = self.seq_design_module(x_noised, aatype_noised, seq_self_cond, None, seq_mask, residue_index)
        seq_logits = self.seq_head(h_V)
        aatype_pred = seq_logits.argmax(dim=-1)  # TODO: need different handling for sampling

        # 2. Sidechain diffusion
        if self.use_scn_diffusion:
            z = self.proj_z(h_V)
            x_bb = x_noised[..., rc.bb_idxs, :]  # TODO: make sure this matches MPNN augment_eps
            x1_scn_pred, scn_diffusion_aux = self.scn_diffusion_module.sidechain_diffusion(
                z,
                aatype_pred,
                x_bb,
                seq_mask=seq_mask,
                residue_index=residue_index,
                aux_inputs=aux_inputs,
                is_sampling=is_sampling
            )
            aux_preds["scn_diffusion_aux"] = scn_diffusion_aux

            if is_sampling:
                # store the predicted sidechain coordinates with known backbone
                x1_pred = cat_bb_scn(x_bb, x1_scn_pred)
            else:
                # during training, the batched x1_scn_pred is in scn_diffusion_aux
                x1_pred = None

        # Outputs
        aux_preds["seq_logits"] = seq_logits
        aux_preds["seq_probs"] = F.softmax(seq_logits, dim=-1)
        return x1_pred, aatype_pred, aux_preds
