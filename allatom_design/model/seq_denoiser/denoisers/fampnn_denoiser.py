from typing import Any, Dict, List, Optional, Tuple

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
from allatom_design.model.seq_denoiser.denoisers.seq_design.fampnn import \
    FAMPNN
from allatom_design.model.seq_denoiser.denoisers.sidechain_diffusion.scn_diffusion_mlp import \
    SidechainDiffusionModule
from esm3.esm.models.esmc import ESMC
from esm3.esm.sdk.api import LogitsConfig
from esm3.esm.utils.sampling import _BatchedESMProteinTensor


ESMC_INFO = {
    "esmc_300m": {"n_layer": 30, "n_channel": 960},
    "esmc_600m": {"n_layer": 36, "n_channel": 1152},
}

class FAMPNNDenoiser(BaseSeqDenoiser):
    def __init__(self,
                 cfg: DictConfig,
                 sigma_data: Tuple[TensorType[(), float], TensorType[(), float]]):
        super().__init__()

        self.cfg = cfg
        self.bb_sigma_data, self.scn_sigma_data = sigma_data
        self.task = cfg.task
        self.use_scn_diffusion = self.task in ["allatom_seq_des", 'scn_pack']

        # Sequence encoding: ESM-C
        self.use_esmc = cfg.get("esm", {}).get("use_esmc", False)
        if self.use_esmc:
            # Load ESM-C model, frozen and in eval mode
            self.esmc_name = cfg.esm.model_name
            self.esmc = ESMC.from_pretrained(self.esmc_name, device=torch.device("cpu")).eval()
            self.vocab = self.esmc.tokenizer.get_vocab()
            for param in self.esmc.parameters():
                param.requires_grad = False

            self.esmc_combine = nn.Parameter(torch.zeros(ESMC_INFO[self.esmc_name]["n_layer"] + 1))  # hidden states + last layer embeddings
            self.esmc_mlp = nn.Sequential(
                nn.LayerNorm(ESMC_INFO[self.esmc_name]["n_channel"]),
                nn.Linear(ESMC_INFO[self.esmc_name]["n_channel"], cfg.fampnn.n_channel),
                nn.ReLU(),
                nn.Linear(cfg.fampnn.n_channel, cfg.fampnn.n_channel),
            )

            # Build lookup tables
            af_to_esm_idx = {}
            esm_to_af_idx = {}

            for aa, af_idx in rc.restype_order_with_x.items():
                if aa == "X":
                    # ESM-C does not use unknowns as mask tokens
                    token_id = self.esmc.tokenizer.mask_token_id
                else:
                    token_id = self.esmc.tokenizer.convert_tokens_to_ids(aa)
                af_to_esm_idx[af_idx] = token_id
                esm_to_af_idx[token_id] = af_idx

            self.register_buffer("af_to_esm", torch.zeros(len(af_to_esm_idx), dtype=torch.long, requires_grad=False))
            for i in range(len(af_to_esm_idx)):
                self.af_to_esm[i] = af_to_esm_idx[i]

            self.register_buffer("esm_to_af", torch.full((max(esm_to_af_idx.keys()) + 1, ), fill_value=-1, dtype=torch.long, requires_grad=False))
            for i in esm_to_af_idx.keys():
                self.esm_to_af[i] = esm_to_af_idx[i]

        # Sequence design model: FAMPNN
        self.seq_design_module = FAMPNN(getattr(cfg, "fampnn", getattr(cfg, "minimpnn", None)))  # backwards compatibility

        # Sidechain diffusion head
        if self.use_scn_diffusion:
            self.scn_diffusion_module = SidechainDiffusionModule(cfg.scn_diffusion_module, self.scn_sigma_data)


    def forward(self,
                x_noised: TensorType["b n a 3", float],
                aatype_noised: TensorType["b n", int],
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

        # Drop residue index for certain proteins in the batch
        if aux_inputs.get("drop_residx", None) is not None:
            residue_index = torch.where(aux_inputs["drop_residx"].unsqueeze(-1), torch.zeros_like(residue_index), residue_index)

        # Sequence encoding
        esmc_embed = None
        if self.use_esmc:
            # make sure to apply mask
            with torch.no_grad():
                protein_tensor = self.af2_to_esmc(aatype_noised, seq_mask)
                logits_output = self.esmc.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True))
            esmc_embed = torch.cat([logits_output.hidden_states, logits_output.embeddings.unsqueeze(0)], dim=0)
            esmc_embed = esmc_embed[:, :, 1:-1]  # remove CLS (first) and EOS (last, but possibly pad)

            # preprocess ESM sequence embedding
            esmc_embed = rearrange(esmc_embed, "l b n h -> b n l h")
            esmc_embed = (self.esmc_combine.softmax(0).unsqueeze(0) @ esmc_embed).squeeze(2)
            esmc_embed = self.esmc_mlp(esmc_embed)
            esmc_embed = esmc_embed * seq_mask.unsqueeze(-1)  # zero out padding & EOS of shorter sequences

        # Sequence design
        seq_logits, mpnn_feature_dict = self.seq_design_module(
            x_noised,
            aatype_noised,
            seq_mask,
            atom_mask_noised,
            residue_index,
            chain_encoding,
            noise=aux_inputs.get("noise", None),
            noise_labels=aux_inputs.get("noise_labels", None),
            h_S_init=esmc_embed)

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

        # Sidechain diffusion
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

        # Return train time modifications
        if self.training:
            aux_preds["X"] = mpnn_feature_dict["X"]  # possibly added noise to X
            aux_preds["noise_labels"] = mpnn_feature_dict["noise_labels"]  # possibly added per-residue noise
            aux_preds["drop_residx"] = aux_inputs.get("drop_residx", None)  # possibly dropped residue indices

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


    def af2_to_esmc(self, aatype: TensorType["b n", int], seq_mask: TensorType["b n", float]) -> _BatchedESMProteinTensor:
        esmc_aatype = self.af_to_esm[aatype]
        pad_token = self.esmc.tokenizer.pad_token_id
        esmc_aatype = torch.where(seq_mask.bool(), esmc_aatype, pad_token)  # mask out padding

        # handle CLS and EOS
        esmc_aatype = F.pad(esmc_aatype, (1, 1), value=pad_token)  # dummy for CLS and EOS
        lengths = seq_mask.sum(dim=-1).long()
        esmc_aatype[:, 0] = self.esmc.tokenizer.cls_token_id
        esmc_aatype[torch.arange(len(lengths)), lengths + 1] = self.esmc.tokenizer.eos_token_id

        return _BatchedESMProteinTensor(esmc_aatype)
