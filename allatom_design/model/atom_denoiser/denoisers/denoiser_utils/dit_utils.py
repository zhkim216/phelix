from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import use_fused_attn
from timm.models.vision_transformer import Mlp
from torch.jit import Final
from torchtyping import TensorType

import allatom_design.model.atom_denoiser.denoisers.pos_embed.rotary_embedding_torch as rope

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_dropout: float, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=mlp_dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self,
                x,
                c: Union[
                    TensorType["b h", float],  # per-sequence conditioning
                    TensorType["b n h", float]  # per-token conditioning
                    ],
                residx: TensorType["b n", float],
                attn_mask: TensorType["b n n", float],
                attn_bias: Optional[TensorType["b n n", float]],
                per_token_conditioning: bool = False,  # whether c is per-token or per-sequence
                rope_mask: TensorType["b n", float] | None = None
                ):
        if not per_token_conditioning:
            assert c.dim() == 2, "Per-sequence conditioning requires shape [B, H] for c"
            c = c.unsqueeze(1)
        assert c.dim() == 3

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), residx=residx, attn_mask=attn_mask,
                                     attn_bias=attn_bias, rope_mask=rope_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self,
                x,
                c: Union[
                    TensorType["b h", float],  # per-sequence conditioning
                    TensorType["b n h", float]  # per-token conditioning
                    ],
                per_token_conditioning: bool = False  # whether c is per-token or per-sequence
        ):
        if not per_token_conditioning:
            assert c.dim() == 2, "Per-sequence conditioning requires shape [B, H] for c"
            c = c.unsqueeze(1)
        assert c.dim() == 3

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Attention(nn.Module):
    """
    Adapated from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py to deal with attention masking.
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            inf: float = 1e9,
            rotary_emb: Optional[rope.RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.inf = inf  # for masked attention
        self.rotary_emb = rotary_emb


    def forward(self,
                x: torch.Tensor,
                residx: TensorType["b n", float],
                attn_mask: TensorType["b h n n", float],
                attn_bias: Optional[TensorType["b h n n", float]],
                rope_mask: TensorType["b n", float] | None
                ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply RoPE
        if self.rotary_emb is not None:
            if rope_mask is not None:
                # Only apply RoPE to non-rope-masked tokens
                q = torch.where(rope_mask[:, None, :, None].bool(),
                                self.rotary_emb.rotate_queries_or_keys(q, residx), q)
                k = torch.where(rope_mask[:, None, :, None].bool(),
                                self.rotary_emb.rotate_queries_or_keys(k, residx), k)
            else:
                q = self.rotary_emb.rotate_queries_or_keys(q, residx)
                k = self.rotary_emb.rotate_queries_or_keys(k, residx)

        # Attention bias / mask
        if attn_bias is None:
            attn_bias = torch.zeros_like(attn_mask)
        attn_bias = torch.where(attn_mask.bool(), attn_bias, -self.inf)

        # Compute attention
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# rmsnorm
# https://github.com/lucidrains/mmdit/blob/main/mmdit/mmdit_pytorch.py
class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale


def modulate(x: TensorType["b n h"], shift: TensorType["b n h"], scale: TensorType["b n h"]):
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        self.has_unconditional_token = use_cfg_embedding  # used externally for default conditioning token settings


    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DenoisingMLPBlock(nn.Module):
    """
    MLP block with adaptive layer norm zero (adaLN-Zero) conditioning. Basically a DiT block, but without attention.
    """
    def __init__(self, hidden_size, mlp_dropout: float, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=mlp_dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

    def forward(self, x, c: TensorType["b n h", float],  # per-token conditioning
                ):
        assert c.dim() == 3
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = x + gate_mlp * self.mlp(modulate(self.norm1(x), shift_mlp, scale_mlp))
        return x
