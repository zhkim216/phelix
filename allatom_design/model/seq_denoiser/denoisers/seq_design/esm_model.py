import esm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class ESMWrapper(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Wrapper around ESM model to return sequence embeddings. Code is based on:
        https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/esmfold.py
        """
        super().__init__()

        self.cfg = cfg
        c_s = cfg.c_s

        self.esm, self.esm_dict = esm_registry.get(cfg.esm_type)()
        self.esm.requires_grad_(False)
        # self.esm.half()  # we train with bf16, so we shouldn't need to half the model

        self.esm_feats = self.esm.embed_dim
        self.esm_attns = self.esm.num_layers * self.esm.attention_heads
        self.register_buffer("af2_to_esm", ESMWrapper._af2_to_esm(self.esm_dict))
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm.num_layers + 1))
        self.embedding = nn.Embedding(self.cfg.n_tokens_embed, c_s, padding_idx=0)
        self.esm_s_mlp = nn.Sequential(
            nn.LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )


    def forward(
        self,
        aatype_noised: TensorType["b n", int],
        seq_mask: TensorType["b n", float],
        residue_index: TensorType["b n", int],
        mlm_mask: TensorType["b n", float],
    ):
        """Runs a forward pass given input tokens.

        Args:
            aatype_noised: Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            seq_mask: Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residue_index: Residue indices of amino acids.
            mlm_mask: MLM mask on the input sequence, 1 denotes unmasked and 0 denotes masked
        """
        esm_aatype = self._af2_idx_to_esm_idx(aatype_noised, seq_mask)
        esm_aatype = self._mask_inputs_to_esm(esm_aatype, seq_mask, mlm_mask)

        # ESM2 doesn't use RoPE on residx?
        # Also, they seem to prepend BOS and append EOS tokens to every example regardless of cropping -- discrepancy with their supplement
        # https://github.com/facebookresearch/esm/issues/299
        # The pretrained ESM model seems to NaN out when neither BOS/EOS are provided
        esm_s, _ = self._compute_language_model_representations(esm_aatype)
        esm_s = esm_s.detach()

        # preprocess ESM sequence embedding
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.esm_s_mlp(esm_s)
        s_s_0 += self.embedding(aatype_noised)

        return s_s_0


    def _compute_language_model_representations(
        self, esmaa: torch.Tensor
    ) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=False,
        )
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        esm_z = None
        return esm_s, esm_z

    def _mask_inputs_to_esm(self,
                            esmaa: TensorType["b n", int],
                            seq_mask: TensorType["b n", float],
                            mlm_mask: TensorType["b n", float]):
        """
        Mask nonpad positions where mlm_mask is 0.
        """
        new_esmaa = esmaa.clone()
        new_esmaa[(mlm_mask == 0) & (seq_mask == 1)] = self.esm_dict.mask_idx
        return new_esmaa

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]


    @staticmethod
    def _af2_to_esm(d: esm.Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [
            d.get_idx(v) for v in rc.restypes_with_x
        ]
        return torch.tensor(esm_reorder)


esm_registry = {
    "esm2_8M": esm.pretrained.esm2_t6_8M_UR50D,
    "esm2_35M": esm.pretrained.esm2_t12_35M_UR50D,
    "esm2_150M": esm.pretrained.esm2_t30_150M_UR50D,
    "esm2_650M": esm.pretrained.esm2_t33_650M_UR50D,
    "esm2_3B": esm.pretrained.esm2_t36_3B_UR50D,
    "esm2_15B": esm.pretrained.esm2_t48_15B_UR50D,
}