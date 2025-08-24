from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.const as const
import allatom_design.model.seq_denoiser.denoisers.seq_design.potts as potts
from allatom_design.data.data import to
# from allatom_design.data.feature.feature_utils import slice_feats
import allatom_design.data.const as const
from allatom_design.model.seq_denoiser.denoisers.denoiser import \
    BaseSeqDenoiser
from allatom_design.model.seq_denoiser.denoisers.seq_design.atom_mpnn import \
    AtomMPNN
from chroma.layers import complexity
import copy


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
        seq_logits, mpnn_feats = self.atom_mpnn(batch, is_sampling)

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

        Ensures that the conditioning masks only contain non-pad, resolved entries.
        Also, updates batch (in place) with:
        - atomwise_token_idx: Tensor["b n_atoms", int]: index of the token that the atom belongs to, 0 for pad atoms
        - atomwise_seq_cond_mask: Tensor["b n_atoms", float]: 1 if the atom is part of an unmasked residue type, or 0 otherwise
        - token_exists_mask: Tensor["b n_tokens", float]: 1 if there exists any unmasked atom in the token, or 0 otherwise
        """
        # Ensure the conditioning masks only contain non-pad, resolved entries
        batch["seq_cond_mask"] = batch["seq_cond_mask"] * batch["token_pad_mask"] * batch["token_resolved_mask"]
        batch["atom_cond_mask"] = batch["atom_cond_mask"] * batch["atom_pad_mask"] * batch["atom_resolved_mask"]

        # Create atom-level mask which is 1 if the atom is part of an unmasked residue type, or 0 otherwise
        batch["atomwise_seq_cond_mask"] = batch["seq_cond_mask"].gather(dim=-1, index=batch["atom_to_token_map"])  # [b, n_atoms]
        batch["atomwise_seq_cond_mask"] = batch["atomwise_seq_cond_mask"] * batch["atom_pad_mask"]  # re-mask out pad atoms, since atom_to_token_map is 0 for pad atoms

        # Build mask for which tokens to include in the token-level grpah
        ## ensure center atom is present, since graph nodes are the center atom
        batch["token_exists_mask"] = batch["token_resolved_mask"].float()  # [b, n_tokens], "whether the token exists in the residue-level graph"

        ## sometimes, it's helpful to mask out certain tokens from the graph (e.g. for protein-only design)
        token_exists_override = batch.get("token_exists_override", torch.ones_like(batch["token_exists_mask"]))
        batch["token_exists_mask"] = batch["token_exists_mask"] * token_exists_override

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
        B, N_atoms = batch["atom_pad_mask"].shape
        N_tokens = batch["token_pad_mask"].shape[1]
        device = batch["atom_pad_mask"].device

        if self.per_residue_eps:
            # per-residue noise. Unlike Cho et al., we sample noise stds from a uniform distribution and apply different noise to each atom in a residue
            # randomly sample noise labels
            noise_labels = torch.rand((B, N_tokens), device=device) * self.augment_eps  # sample std for each residue from uniform [0, augment_eps]
            atomwise_noise_labels = noise_labels.gather(dim=-1, index=batch["atom_to_token_map"]) * batch["atom_pad_mask"] # [b, n_atoms]
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


    def potts_sample(self, batch: dict[str, TensorType["b ..."]], sampling_inputs: dict[str, Any]):
        """
        Potts sampling for sequence design.

        Returns:
            output_feats: list[dict[str, TensorType["b ..."]]]: list of length (n_samples_per_pdb) of output features for each sample
            aux: dict[str, Any]: auxiliary outputs
        """
        aux = {}

        # If specified, add noise to inputs
        noise_std = sampling_inputs["potts_sampling_cfg"].get("noise_std", 0.0)
        if noise_std > 0:
            batch["coords"] = batch["coords"] + torch.randn_like(batch["coords"]) * noise_std
            batch["coords"] = batch["coords"] * batch["atom_pad_mask"].unsqueeze(-1) * batch["atom_resolved_mask"].unsqueeze(-1)

        # If specified, condition on sequence only in the potts model
        batch["seq_cond_mask_potts"] = batch["seq_cond_mask"].clone()
        if sampling_inputs["potts_sampling_cfg"].get("potts_only_cond", False):
            print("Conditioning on sequence only in the potts model")
            batch["seq_cond_mask"] = torch.zeros_like(batch["seq_cond_mask"])  # zero out model-level sequence conditioning mask

        # Compute potts parameters
        potts_decoder_aux, batch, sampling_inputs = self.compute_potts_params(batch, sampling_inputs,
                                                                              use_msa_potts=sampling_inputs["potts_sampling_cfg"].get("use_msa_potts", False))
        aux["potts_decoder_aux"] = to(potts_decoder_aux, "cpu")

        # Set up Potts sampling
        potts_sampling_cfg = sampling_inputs["potts_sampling_cfg"]
        regularization = potts_sampling_cfg["regularization"]
        potts_sweeps = potts_sampling_cfg["potts_sweeps"]
        potts_proposal = potts_sampling_cfg["potts_proposal"]
        potts_temperature = potts_sampling_cfg["potts_temperature"]
        rejection_step = potts_sampling_cfg.get("rejection_step", potts_proposal == "chromatic")

        B, N, _ = batch["res_type"].shape
        logits_init = torch.zeros((B, N, const.AF3_SEQUENCE_ENCODING.n_tokens), device=batch["res_type"].device).float()

        # Handle banned amino acids and aatype restrictions
        ban_S = {"X"}
        omit_aas = sampling_inputs.get("omit_aas", None)
        if omit_aas is not None:
            ban_S = ban_S | set(omit_aas)
        ban_S = [const.token_ids[const.prot_letter_to_token[aa]] for aa in ban_S]
        ban_S.extend([const.token_ids[x] for x in const.tokens if x not in const.prot_only_tokens])  # ban all non-protein tokens

        # Initialize random sequence and sampling masks
        mask_sample = (1 - batch["seq_cond_mask_potts"]) * batch["token_pad_mask"]  # 1 where we can sample, 0 where we can't
        mask_sample, _, S_init = potts.init_sampling_masks(
            logits_init, mask_sample=mask_sample, S=batch["res_type"].argmax(dim=-1), ban_S=ban_S, pos_restrict_aatype=sampling_inputs.get("pos_restrict_aatype", None)
        )

        # Complexity regularization
        penalty_func = None
        mask_ij_coloring = None
        edge_idx_coloring = None
        if regularization == "LCP":
            C_complexity = batch["asym_id"] - torch.min(batch["asym_id"]) + 1  # renumber asym_id to have min value of 1
            C_complexity = C_complexity * batch["token_pad_mask"] * batch["token_exists_mask"]  # mask out pad tokens and tokens that don't exist in the graph
            penalty_func = lambda _S: complexity.complexity_lcp(_S, C_complexity)

        S = []  # keep track of sequences for each sample
        aux["U"] = []  # keep track of energies for each sample

        # Design sequences
        for _ in tqdm(range(sampling_inputs["num_seqs_per_pdb"]), desc="Sampling sequences", leave=False):
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
            for chain_type, unk_token_id in const.unk_token_ids.items():
                chain_type_id = const.chain_type_ids[chain_type]
                unk_mask = (~batch["token_exists_mask"].bool()) & (batch["mol_type"] == chain_type_id)
                S_sample[unk_mask] = unk_token_id

            aux["U"].append(U_sample.cpu())
            S.append(S_sample.cpu())

        batch = to(batch, device="cpu")

        # Thread sequences onto original batch
        output_feats = []
        aux["input_res_type"] = []  # keep track of original res_types
        for si in range(len(S)):
            feats_si = copy.deepcopy(batch)
            feats_si["res_type"] = torch.where(feats_si["seq_cond_mask"][..., None].bool(),
                                               feats_si["res_type"],
                                               F.one_hot(S[si], num_classes=const.AF3_SEQUENCE_ENCODING.n_tokens))
            feats_si["coords"] = feats_si["coords"] * feats_si["atom_cond_mask"].unsqueeze(-1)
            feats_si["atom_resolved_mask"] = feats_si["atom_resolved_mask"] * feats_si["atom_cond_mask"]
            output_feats.append(feats_si)

            # Return input res_type
            aux["input_res_type"].append(batch["res_type"].cpu())

        return output_feats, aux


    def compute_potts_params(self, batch: dict[str, TensorType["b ..."]],
                             sampling_inputs: dict[str, Any],
                             use_msa_potts: bool = False) -> tuple[dict[str, TensorType["b ..."]], dict[str, TensorType["b ..."]], dict[str, Any]]:
        """
        Run model and collect potts parameters over a batch of samples.

        If "tied_sampling_ids" is in batch, we will aggregate potts parameters across tied groups and slice batch to representative elements.

        Returns:
            potts_decoder_aux: dict[str, TensorType["b ..."]]: potts parameters
            batch: dict[str, TensorType["b ..."]]: batch with token_exists_mask added
            sampling_inputs: dict[str, Any]: sampling inputs with pos_restrict_aatype sliced to representative elements
        """
        subbatch_size = sampling_inputs["batch_size"]
        B = batch["res_type"].shape[0]

        # Run model and collect potts parameters
        potts_decoder_aux = {}  # potts parameters
        token_exists_mask = []  # keep track of the tokens that exist in the graph
        for bi in tqdm(range(0, B, subbatch_size), desc="Computing potts parameters", leave=False):
            subbatch = slice_feats(batch, slice(bi, bi + subbatch_size))

            _, aux_preds_i = self(subbatch, is_sampling=True, sampling_inputs=sampling_inputs)

            for k, v in aux_preds_i["potts_decoder_aux"].items():
                potts_decoder_aux.setdefault(k, []).append(v)
            token_exists_mask.append(aux_preds_i["token_exists_mask"])
        potts_decoder_aux = {k: torch.cat(v, dim=0) for k, v in potts_decoder_aux.items()}
        token_exists_mask = torch.cat(token_exists_mask, dim=0)
        batch["token_exists_mask"] = token_exists_mask  # store in batch for downstream use

        # If using MSA potts, we use h_msa and J_msa instead of h and J
        if use_msa_potts:
            potts_decoder_aux["h"] = potts_decoder_aux.pop("h_msa")
            potts_decoder_aux["J"] = potts_decoder_aux.pop("J_msa")

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
