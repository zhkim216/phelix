from collections import defaultdict
from functools import partial
from typing import Dict

import boltz.model.layers.initialize as init
import torch
import torch.nn as nn
from boltz.model.modules.encoders import get_indexing_matrix, single_to_keys
from boltz.model.modules.utils import LinearNoBias

from allatom_design.model.atom_denoiser.denoisers.denoiser_utils.boltz_transformers import \
    AtomTransformer


class MotifEmbedder(nn.Module):
    """Motif embedder adapted from Boltz InputEmbedder."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        atom_encoder_depth: int,
        atom_encoder_heads: int,
        no_atom_encoder: bool = False,
    ) -> None:
        """Initialize the input embedder.

        Parameters
        ----------
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        token_s : int
            The single token representation dimension.
        atoms_per_window_queries : int
            The number of atoms per window for queries.
        atoms_per_window_keys : int
            The number of atoms per window for keys.
        atom_feature_dim : int
            The atom feature dimension.
        atom_encoder_depth : int
            The atom encoder depth.
        atom_encoder_heads : int
            The atom encoder heads.
        no_atom_encoder : bool, optional
            Whether to use the atom encoder, by default False

        """
        super().__init__()
        self.token_s = token_s
        self.no_atom_encoder = no_atom_encoder
        self.to_motif_embed_1d = LinearNoBias(token_s + 33 + 4, token_s)  # token_s + 33 restypes + 4 pocket features

        if not no_atom_encoder:
            self.atom_attention_encoder = AtomAttentionEncoder(
                atom_s=atom_s,
                atom_z=atom_z,
                token_s=token_s,
                atoms_per_window_queries=atoms_per_window_queries,
                atoms_per_window_keys=atoms_per_window_keys,
                atom_feature_dim=atom_feature_dim,
                atom_encoder_depth=atom_encoder_depth,
                atom_encoder_heads=atom_encoder_heads,
            )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        feats : Dict[str, Tensor]
            Input features

        Returns
        -------
        Tensor
            The embedded tokens.

        """
        # Load relevant features
        res_type = feats["res_type"]
        pocket_feature = feats["pocket_feature"]

        # Compute input embedding
        if self.no_atom_encoder:
            a = torch.zeros(
                (res_type.shape[0], res_type.shape[1], self.token_s),
                device=res_type.device,
            )
        else:
            a, _, _, _, _ = self.atom_attention_encoder(feats)
        s = torch.cat([a, res_type, pocket_feature], dim=-1)
        motif_embed_1d = self.to_motif_embed_1d(s)
        return motif_embed_1d


class AtomAttentionEncoder(nn.Module):
    """Modified atom attention encoder adapted from Boltz."""

    def __init__(
        self,
        atom_s,
        atom_z,
        token_s,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_feature_dim,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
    ):
        """Initialize the atom attention encoder.

        Parameters
        ----------
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        token_s : int
            The single representation dimension.
        atoms_per_window_queries : int
            The number of atoms per window for queries.
        atoms_per_window_keys : int
            The number of atoms per window for keys.
        atom_feature_dim : int
            The atom feature dimension.
        atom_encoder_depth : int, optional
            The number of transformer layers, by default 3.
        atom_encoder_heads : int, optional
            The number of transformer heads, by default 4.
        """
        super().__init__()

        self.embed_atom_features = LinearNoBias(atom_feature_dim, atom_s)
        self.embed_atompair_ref_pos = LinearNoBias(3, atom_z)
        self.embed_atompair_ref_dist = LinearNoBias(1, atom_z)
        self.embed_atompair_mask = LinearNoBias(1, atom_z)
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys

        # Embed input coords r
        self.r_to_q_trans = LinearNoBias(4, atom_s)  # xyz + mask = 4
        init.final_init_(self.r_to_q_trans.weight)

        self.c_to_p_trans_k = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )
        init.final_init_(self.c_to_p_trans_k[1].weight)

        self.c_to_p_trans_q = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )
        init.final_init_(self.c_to_p_trans_q[1].weight)

        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
        )
        init.final_init_(self.p_mlp[5].weight)

        self.atom_encoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            dim_pairwise=atom_z,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
        )

        self.atom_to_token_trans = nn.Sequential(
            LinearNoBias(atom_s, token_s),
            nn.ReLU(),
        )


    def forward(
        self,
        feats,
    ):
        B, N, _ = feats["ref_pos"].shape
        atom_mask = feats["atom_pad_mask"].bool()

        atom_ref_pos = feats["ref_pos"]
        atom_uid = feats["ref_space_uid"]
        atom_feats = torch.cat(
            [
                atom_ref_pos,
                feats["ref_charge"].unsqueeze(-1),
                feats["atom_pad_mask"].unsqueeze(-1),
                feats["ref_element"],
                feats["ref_atom_name_chars"].reshape(B, N, 4 * 64),
            ],
            dim=-1,
        )

        c = self.embed_atom_features(atom_feats)

        # NOTE: we are already creating the windows to make it more efficient
        W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
        B, N = c.shape[:2]
        K = N // W
        keys_indexing_matrix = get_indexing_matrix(K, W, H, c.device)
        to_keys = partial(
            single_to_keys, indexing_matrix=keys_indexing_matrix, W=W, H=H
        )

        atom_ref_pos_queries = atom_ref_pos.view(B, K, W, 1, 3)
        atom_ref_pos_keys = to_keys(atom_ref_pos).view(B, K, 1, H, 3)

        d = atom_ref_pos_keys - atom_ref_pos_queries
        d_norm = torch.sum(d * d, dim=-1, keepdim=True)
        d_norm = 1 / (1 + d_norm)

        atom_mask_queries = atom_mask.view(B, K, W, 1)
        atom_mask_keys = (
            to_keys(atom_mask.unsqueeze(-1).float()).view(B, K, 1, H).bool()
        )
        atom_uid_queries = atom_uid.view(B, K, W, 1)
        atom_uid_keys = (
            to_keys(atom_uid.unsqueeze(-1).float()).view(B, K, 1, H).long()
        )
        v = (
            (
                atom_mask_queries
                & atom_mask_keys
                & (atom_uid_queries == atom_uid_keys)
            )
            .float()
            .unsqueeze(-1)
        )

        p = self.embed_atompair_ref_pos(d) * v
        p = p + self.embed_atompair_ref_dist(d_norm) * v
        p = p + self.embed_atompair_mask(v) * v

        q = c

        p = p + self.c_to_p_trans_q(c.view(B, K, W, 1, c.shape[-1]))
        p = p + self.c_to_p_trans_k(to_keys(c).view(B, K, 1, H, c.shape[-1]))
        p = p + self.p_mlp(p)

        # Embed motif coords and mask
        r_mask = feats["motif_atom_mask"].unsqueeze(-1)
        r = feats["motif_coords"] * r_mask
        r_input = torch.cat(
            [r, r_mask],
            dim=-1,
        )
        r_to_q = self.r_to_q_trans(r_input)
        q = q + r_to_q

        q = self.atom_encoder(
            q=q,
            mask=atom_mask.float(),
            c=c,
            p=p,
            to_keys=to_keys,
        )

        q_to_a = self.atom_to_token_trans(q)
        atom_to_token = feats["atom_to_token"].float()
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        return a, q, c, p, to_keys
