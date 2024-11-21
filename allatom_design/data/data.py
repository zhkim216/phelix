from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchtyping import TensorType

import openfold.data.data_transforms as data_transforms
from allatom_design.data import protein
from allatom_design.data import residue_constants as rc
from openfold.utils.feats import atom14_to_atom37
from pathlib import Path
from openfold.utils.rigid_utils import Rigid, Rotation
import subprocess
from typing import Tuple, Union
import math


def load_feats_from_pdb(pdb, chain_residx_gap: int, max_conformers: int = 1):
    """
    Load model input features from a PDB file or mmcif file.
    - chain_residx_gap: Gap to add between residue indices in different chains.
    - max_conformers: Handle disordered atoms, max number of altlocs to store. If > 1, returns coords with shape [seqlen, num_atoms, max_conformers, 3]
    """
    feats = {}
    protein_obj = protein.read_pdb(pdb, max_conformers=max_conformers)
    for k, v in vars(protein_obj).items():
        feats[k] = torch.Tensor(v)

    feats["all_atom_positions"] = feats.pop("atom_positions")
    feats["all_atom_mask"] = feats.pop("atom_mask")

    feats["aatype"] = feats["aatype"].long()

    # Renumber residue indices; add gap for PDBs with multiple chains
    if chain_residx_gap is not None:
        raise NotImplementedError("Currently not supporting multiple chains, since this may require renumbering residues across chains with a scheme to handle missing residues within chains.")
        # feats["residue_index"] = renumber_and_add_chain_gap(feats["residue_index"], feats["chain_index"], chain_residx_gap=chain_residx_gap)

    # Add one-hot encoding of amino acid types
    feats["target_feat"] = F.one_hot(feats["aatype"], num_classes=len(rc.restypes_with_x)).float()

    # Add AF2 features, uncomment if needed
    feats = data_transforms.make_seq_mask(feats)
    # feats = data_transforms.make_atom14_masks(feats)
    # feats = data_transforms.make_atom14_positions(feats)
    # feats = data_transforms.atom37_to_frames(feats)
    # feats = data_transforms.atom37_to_torsion_angles("")(feats)
    # feats = data_transforms.make_pseudo_beta("")(feats)
    # feats = data_transforms.get_backbone_frames(feats)
    # feats = data_transforms.get_chi_angles(feats)

    # Handle the distinction between missing atoms and ghost atoms in the atom mask
    ghost_atom_mask = 1 - torch.tensor(rc.restype_atom37_mask)[feats["aatype"]]  # 1 for atoms that are not in the residue type; ghost atoms
    if max_conformers > 1:
        ghost_atom_mask = rearrange(ghost_atom_mask, "n a -> n 1 a").expand(-1, max_conformers, -1)  # [n, c, a]

    missing_atom_mask = (1 - feats["all_atom_mask"]) * (1 - ghost_atom_mask)  # 1 for atoms that are missing in the PDB file; missing if not in atom_mask but not a ghost atom

    feats["ghost_atom_mask"] = ghost_atom_mask  # [n, a] or [n, c, a]
    feats["missing_atom_mask"] = missing_atom_mask  # [n, a] or [n, c, a]
    feats["res_b_factors"] = torch.sum(feats["b_factors"], dim = -1) / torch.sum((1 - ghost_atom_mask), dim = -1)
    return feats


def aa_to_bb_feats(feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert features loaded from a PDB file to backbone-only features.
    """
    bb_feats = {}
    gly_aatype = rc.restype_order["G"]

    # Replace aatype with GLY and get only backbone atoms
    bb_feats["aatype"] = torch.full_like(feats["aatype"], fill_value=gly_aatype)
    bb_feats["all_atom_positions"] = feats["all_atom_positions"][:, rc.bb_idxs]
    bb_feats["all_atom_mask"] = feats["all_atom_mask"][:, rc.bb_idxs]
    bb_feats["seq_mask"] = feats["seq_mask"]
    bb_feats["residue_index"] = feats["residue_index"]
    bb_feats["chain_index"] = feats["chain_index"]

    bb_feats["ghost_atom_mask"] = feats["ghost_atom_mask"][:, rc.bb_idxs]
    bb_feats["missing_atom_mask"] = feats["missing_atom_mask"][:, rc.bb_idxs]

    bb_feats["target_feat"] = F.one_hot(bb_feats["aatype"], num_classes=len(rc.restypes_with_x)).float()
    return bb_feats


def renumber_and_add_chain_gap(residue_index: TensorType["n"],
                               chain_index: TensorType["n"],
                               chain_residx_gap: int = 200) -> TensorType["n"]:
    """
    Renumber residue indices to start from 1 and add a residue index gap between chains.
    e.g. if chain A has 5 residues and chain B has 3 residues,
    with chain_residx_gap=200, the residue indices for chain B will be 206, 207, 208.
    """
    # First, make residue indices are linearly ordered across chains
    residue_index = torch.arange(1, residue_index.shape[0] + 1)

    # Now add a gap to the residue index for each chain break
    residue_index = residue_index + chain_residx_gap * chain_index

    return residue_index


def make_fixed_size_1d(data: TensorType["n ..."], fixed_size: int, start_idx: int):
    data_len = data.shape[0]
    if data_len > fixed_size:
        new_data = data[start_idx : (start_idx + fixed_size)]
    if data_len <= fixed_size:
        pad_size = fixed_size - data_len
        extra_shape = data.shape[1:]
        new_data = torch.cat([data, torch.zeros(pad_size, *extra_shape)], 0)
    return new_data


def dgram_from_positions(
    pos: torch.Tensor,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    no_bins: float = 39,
    inf: float = 1e8,
):
    dgram = torch.sum(
        (pos[..., None, :] - pos[..., None, :, :]) ** 2, dim=-1, keepdim=True
    )
    lower = torch.linspace(min_bin, max_bin, no_bins, device=pos.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)

    return dgram


def build_struct_pair_feat(
    batch, min_bin, max_bin, no_bins, inf=1e8
):
    """
    Adapted from https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/feats.py#L110
    """
    mask = batch["pseudo_beta_mask"]
    mask_2d = mask[..., None] * mask[..., None, :]

    # Compute distogram (this seems to differ slightly from Alg. 5)
    pb = batch["pseudo_beta"]
    dgram = dgram_from_positions(pb, min_bin, max_bin, no_bins, inf)

    to_concat = [dgram, mask_2d[..., None]]

    act = torch.cat(to_concat, dim=-1)
    act = act * mask_2d[..., None]

    return act


def atom14_aatype_to_atom37(atom14_pos: TensorType["b n 14 3", float],
                            aatype: TensorType["b n", int]
                            ) -> TensorType["b n 37 3", float]:
    feats = {}
    feats["aatype"] = aatype
    feats = data_transforms.make_atom14_masks(feats)
    return atom14_to_atom37(atom14_pos, feats)


def torch_kabsch(a: TensorType["b n x"],
                 b: TensorType["b n x"]
                 ) -> TensorType["b x x"]:
    """
    get alignment matrix for two sets of coordinates using PyTorch

    adapted from: https://github.com/sokrypton/ColabDesign/blob/ed4b01354928b60cd1347f570e9b248f78f11c6d/colabdesign/shared/protein.py#L128
    """
    with torch.autocast(device_type=a.device.type, enabled=False):
        ab = a.transpose(-1, -2) @ b
        u, s, vh = torch.linalg.svd(ab, full_matrices=False)
        flip = torch.det(u @ vh) < 0
        u_ = torch.where(flip, -u[..., -1].T, u[..., -1].T).T
    u = torch.cat([u[..., :-1], u_[..., None]], dim=-1)
    return u @ vh


def torch_rmsd_weighted(a: TensorType["b n x", float],
                        b: TensorType["b n x", float],
                        weights: Optional[TensorType["b n", float]],
                        return_aligned: bool = False
                        ) -> TensorType["b", float]:
    """
    Compute weighted RMSD of coordinates after weighted alignment. Batched.

    For masked RMSD, set weights to 0 for masked atoms.

    Aligns a to b using Kabsch algorithm, then computes RMSD.
    If return_aligned is True, returns the aligned structures as well.

    Adapted from: https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py#L445
    """
    if weights is None:
        weights = torch.ones(a.shape[:-1], device=a.device, dtype=a.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize weights

    # Align
    W = weights[..., None]
    a_mu = (a * W).sum(dim=-2, keepdim=True)
    b_mu = (b * W).sum(dim=-2, keepdim=True)

    R = torch_kabsch((a - a_mu) * W, b - b_mu)
    aligned_a = (a - a_mu) @ R + b_mu

    weighted_msd = (W * ((aligned_a - b) ** 2)).sum(dim=(-1, -2))
    weighted_rmsd = torch.sqrt(weighted_msd + 1e-8)

    if return_aligned:
        return weighted_rmsd, (aligned_a, b)
    return weighted_rmsd


def atom37_to_torsions_rad(aatype: TensorType["b n", int],
                           coords: TensorType["b n 37 3", float],
                           atom_mask: TensorType["b n 37", float]
                           ) -> Tuple[TensorType["b n 7"], TensorType["b n 7"]]:
    """
    Uses OpenFold's atom37_to_torsion_angles to convert atom37 coordinates to torsion angles in radians.
    """
    feats = data_transforms.atom37_to_torsion_angles("")({"aatype": aatype, "all_atom_positions": coords, "all_atom_mask": atom_mask})

    sin_angles, cos_angles = feats["torsion_angles_sin_cos"][..., 0], feats["torsion_angles_sin_cos"][..., 1]
    torsions_deg = torch.atan2(sin_angles, cos_angles)

    alt_sin_angles, alt_cos_angles = feats["alt_torsion_angles_sin_cos"][..., 0], feats["alt_torsion_angles_sin_cos"][..., 1]
    alt_torsions_deg = torch.atan2(alt_sin_angles, alt_cos_angles)

    torsion_angles_mask = feats["torsion_angles_mask"]
    return torsions_deg, alt_torsions_deg, torsion_angles_mask


def tm_score(a: TensorType["b n a 3"],
             b: TensorType["b n a 3"],
             mask: TensorType["b n a"]
             ) -> TensorType["b", float]:
    """
    Computes the TM-score between two sets of coordinates. Batched.
    a and b must be aligned.
    """
    length = b.shape[1]
    dists = (a - b).pow(2).sum(-1)
    d0 = 1.24 * ((length - 15) ** (1 / 3)) - 1.8
    term = 1 / (1 + ((dists) / (d0**2)))

    term = term * mask
    return term.sum(dim=(-1, -2)) / mask.sum(dim=(-1, -2)).clamp(min=1)


def uniform_rand_rotation(batch_size):
    # Creates a shape (batch_size, 3, 3) rotation matrix uniformly at random in SO(3)
    # Uses quaternionic multiplication to generate independent rotation matrices for each batch
    q = torch.randn(batch_size, 4)
    q /= torch.norm(q, dim=1, keepdim=True)
    rotation = torch.zeros(batch_size,3,3).to(q)
    a, b, c, d = q[:,0], q[:,1], q[:,2], q[:,3]
    rotation[:,0,:] = torch.stack([2*a**2 -1 + 2*b**2,   2*b*c - 2*a*d,        2*b*d + 2*a*c]).T
    rotation[:,1,:] = torch.stack([2*b*c + 2*a*d,        2*a**2 -1 + 2*c**2,   2*c*d - 2*a*b]).T
    rotation[:,2,:] = torch.stack([2*b*d - 2*a*c,        2*c*d + 2*a*b,        2*a**2 -1 + 2*d**2]).T
    return rotation


def center_random_augmentation(coords_in: TensorType["n a 3", float],
                               seq_mask: TensorType["n", float],
                               atom_mask: TensorType["n a", float],
                               missing_atom_mask: TensorType["n a", float],
                               translation_scale=1.0,
                               return_transforms=False
                               ):
    """
    Batched or unbatched.
    Mean center on CA atoms, then apply random rotation and translation.
    Ensures that missing/ghost/padding atoms are set back to 0.

    Inputs:
        - seq_mask: 0 if residue is padding
        - atom_mask: 1 if not ghost and not missing atom, 0 otherwise
        - missing_atom_mask: 1 if atom is missing, 0 if present
    """
    input_dim = coords_in.dim()
    if input_dim == 3:
        # unbatched; add batch dimension
        coords_in = coords_in.unsqueeze(0)
        atom_mask = atom_mask.unsqueeze(0)
        missing_atom_mask = missing_atom_mask.unsqueeze(0)
        seq_mask = seq_mask.unsqueeze(0)

    X = coords_in[:, :, 1:2]  # [b n 1 3]

    # Center coords
    M = (1 - missing_atom_mask[:, :, 1:2]) * seq_mask[:, :, None]  # [b n 1]
    M_sum = M.sum(dim=1, keepdim=True)[..., None]  # [b 1 1 1]
    coords_mean = (X * M[..., None]).sum(dim=1, keepdim=True) / M_sum  # [b 1 1 3]
    coords_in = coords_in - coords_mean

    # Apply random rotation
    random_rot = uniform_rand_rotation(coords_in.shape[0]).to(coords_in.device)
    coords_in = torch.einsum("b n a i, b i j -> b n a j", coords_in, random_rot)

    # Apply random translation
    random_trans = torch.randn_like(coords_mean) * translation_scale
    coords_in = coords_in + random_trans

    # Zero out padding + missing / ghost atoms
    coords_in = coords_in * rearrange(seq_mask, "b n -> b n 1 1")
    coords_in = coords_in * atom_mask[..., None]

    transforms = (coords_mean, random_rot, random_trans)
    if input_dim == 3:
        # unbatched; remove batch dimension
        coords_in = coords_in.squeeze(0)
        transforms = tuple(t.squeeze(0) for t in transforms)

    if return_transforms:
        return coords_in, transforms

    return coords_in


def apply_random_augmentation(coords_in: TensorType["b n a 3", float],
                              transforms: Tuple[TensorType["b 1 1 3", float], TensorType["b 3 3", float], TensorType["b 1 1 3", float]],
                              seq_mask: TensorType["b n", float],
                              atom_mask: TensorType["b n a", float]) -> TensorType["b n a 3", float]:
    """
    Batched or unbatched.

    Given the output transforms of center_random_augmentation, applies the same transformation to a set of coordinates.
    Ensures that missing/ghost/padding atoms are set back to 0.
    """
    input_dim = coords_in.dim()
    if input_dim == 3:
        # unbatched; add batch dimension
        coords_in = coords_in.unsqueeze(0)
        transforms = tuple(t.unsqueeze(0) for t in transforms)

    coords_mean, random_rot, random_trans = transforms

    # Apply transforms
    coords_in = coords_in - coords_mean
    coords_in = torch.einsum("b n a i, b i j -> b n a j", coords_in, random_rot)
    coords_in = coords_in + random_trans

    # Zero out padding + missing / ghost atoms
    coords_in = coords_in * rearrange(seq_mask, "b n -> b n 1 1")
    coords_in = coords_in * atom_mask[..., None]

    if input_dim == 3:
        # unbatched; remove batch dimension
        coords_in = coords_in.squeeze(0)

    return coords_in


def cat_bb_scn(x_bb: TensorType["... a1 3", float],
               x_scn: TensorType["... a2 3", float]) -> TensorType["... a 3", float]:
    """
    Concatenate the bb and scn atoms to their corresponding indices.
    """
    A = x_bb.shape[-2] + x_scn.shape[-2]
    x = torch.zeros(x_bb.shape[:-2] + (A, 3), device=x_bb.device, dtype=x_bb.dtype)
    x[..., rc.bb_idxs, :] = x_bb
    x[..., rc.non_bb_idxs, :] = x_scn
    return x


def stack_aux_traj(aux_traj: List[Dict[str, Any]], dim: int = 1) -> Dict[str, Any]:
    """
    Stacks tensors from a list of dictionaries, recursively handling nested dictionaries.
    """
    stacked = {}
    first_item = aux_traj[0]

    for key, value in first_item.items():
        if isinstance(value, dict):
            sub_dicts = [x[key] for x in aux_traj]
            stacked[key] = stack_aux_traj(sub_dicts, dim=dim)
        else:
            stacked[key] = torch.stack([x[key] for x in aux_traj], dim=dim)

    return stacked


def atom14_aatype_to_atom37(atom14_pos: TensorType["b n 14 3", float],
                            aatype: TensorType["b n", int]) -> TensorType["b n 37 3", float]:
    feats = {}
    feats["aatype"] = aatype
    feats = data_transforms.make_atom14_masks(feats)
    return atom14_to_atom37(atom14_pos, feats)

def get_rc_tensor(rc_np, aatype):
    return torch.tensor(rc_np, device=aatype.device)[aatype]

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]

def atom37_to_atom14(aatype, all_atom_pos):
    """Convert Atom37 positions to Atom14 positions."""
    atom37_mask = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype)

    residx_atom14_to_atom37 = get_rc_tensor(
        rc.RESTYPE_ATOM14_TO_ATOM37, aatype
    )

    no_batch_dims = len(aatype.shape) - 1

    atom14_mask = batched_gather(
        atom37_mask,
        residx_atom14_to_atom37,
        dim=no_batch_dims + 1,
        no_batch_dims=no_batch_dims + 1,
    ).to(all_atom_pos.dtype)

    # create a mask for known groundtruth positions
    atom14_mask *= get_rc_tensor(rc.RESTYPE_ATOM14_MASK_WITH_X, aatype)

    # gather the groundtruth positions
    atom14_positions = batched_gather(
        all_atom_pos,
        residx_atom14_to_atom37,
        dim=no_batch_dims + 1,
        no_batch_dims=no_batch_dims + 1,
    )

    return atom14_positions, atom14_mask


# >>> GRAPH TRANSFORMER UTILS

def extract_ids_topk(X, num_nn = 64):
    # compute displacement vectors
    R = X.unsqueeze(0) - X.unsqueeze(1)

    # compute distance matrix
    D = torch.norm(R, dim=2)

    # mask distances
    D = D + torch.max(D)*(D < 1e-2).float()

    # find nearest neighbors
    knn = min(num_nn, D.shape[0])
    _, ids_topk = torch.topk(D, knn, dim=1, largest=False)

    return ids_topk

def unpack(packed_rep, tgt_shape, mask):
    device, dtype = packed_rep.device, packed_rep.dtype
    out = torch.zeros(tgt_shape, device = device, dtype = dtype)
    out[mask, :] = packed_rep
    return out

def pack(unpacked_rep, tgt_shape, mask):
    return unpacked_rep[mask].reshape(tgt_shape)

def gather_pos_enc(
        ids_topk,
        positional_enc,
        return_zero = False
    ):

    ids_topk_expanded = ids_topk.unsqueeze(2).expand(-1, -1, positional_enc.shape[-1])
    positional_enc_topk = torch.gather(
        positional_enc, dim=1,
        index=ids_topk_expanded
    ).float()

    return positional_enc_topk if not return_zero else torch.zeros_like(ids_topk_expanded)

def get_graph_transformer_inputs(
    X,
    atom14_mask,
    aatype_noised,
    seq_mask,
    chain_encoding,
    max_nn,
    pos_enc,
    attn_bias
):
    #get one hot encoding of atom identities
    atom_indices = get_rc_tensor(rc.RESTYPE_TO_ATOM37_IDX, aatype_noised[seq_mask == 1].flatten())
    atom_indices_packed = atom_indices[atom_indices != -1].flatten()
    q = F.one_hot(atom_indices_packed, num_classes=len(rc.atom_types)).float()
    tot_num_atoms = len(atom_indices_packed)

    #packing X and getting packed mask
    atoms14_mask_no_pad = atom14_mask * seq_mask[:,:, None]
    atom14_mask_packed_no_pad = (atom14_mask[seq_mask == 1] == 1)
    chain_encoding_packed_no_pad = chain_encoding[seq_mask == 1].flatten()
    unmasked_packed_X = pack(
            unpacked_rep = X[seq_mask == 1, ...],
            tgt_shape = (-1, 3),
            mask = atom14_mask_packed_no_pad
        )

    num_atoms_per_example = atoms14_mask_no_pad.sum(dim =(1,2)).long()
    num_residues_per_example = seq_mask.sum(dim = -1).long()
    num_atoms_per_residue = atom14_mask_packed_no_pad.sum(dim = -1).long().to(X.device)
    batch_atom_start_idx = torch.cat((torch.tensor([0], device=q.device), num_atoms_per_example[:-1])).cumsum(dim=0)
    batch_residue_start_idx = torch.cat((torch.tensor([0], device=q.device), num_residues_per_example[:-1])).cumsum(dim=0)
    ids_topk = torch.zeros((tot_num_atoms, max_nn), dtype=torch.long, device=q.device)
    positional_enc_topk = torch.zeros((tot_num_atoms, max_nn, 137), dtype=q.dtype, device=q.device) if (pos_enc or attn_bias) else None

    # Process each batch example
    for atom_start_idx, residue_start_idx, na, nr in zip(batch_atom_start_idx, batch_residue_start_idx, num_atoms_per_example, num_residues_per_example):
        # Extract packed_X and compute ids_topk
        start_a, end_a = int(atom_start_idx), int(atom_start_idx + na)
        start_r, end_r = int(residue_start_idx), int(residue_start_idx + nr)
        packed_X_i = unmasked_packed_X[start_a: end_a, :]
        ids_topk_i = extract_ids_topk(packed_X_i, num_nn = max_nn)

        if (pos_enc or attn_bias):
            num_atoms_per_residue_i = num_atoms_per_residue[start_r: end_r]
            chain_encoding_i = chain_encoding_packed_no_pad[start_r: end_r]
            atom_residue_idx_i = torch.repeat_interleave(
                torch.arange(nr).to(q.device),
                num_atoms_per_residue_i
            )

            atom_chain_enc_i = torch.repeat_interleave(
                chain_encoding_i,
                num_atoms_per_residue_i
            )

            same_res = torch.eq(atom_residue_idx_i.unsqueeze(0), atom_residue_idx_i.unsqueeze(1))
            same_chain = torch.eq(atom_chain_enc_i.unsqueeze(0), atom_chain_enc_i.unsqueeze(1))
            atom_idx_i = torch.arange(na).to(q.device)
            d_atom = torch.clamp(atom_idx_i.unsqueeze(0) - atom_idx_i.unsqueeze(1) + rc.r_max, min = 0, max = 2 * rc.r_max)
            d_atom[~(same_chain|same_res)] = 2 * rc.r_max + 1
            rel_atom_enc = F.one_hot(d_atom, num_classes = 2 * rc.r_max + 2)
            d_chain = torch.clamp(atom_chain_enc_i.unsqueeze(0) - atom_chain_enc_i.unsqueeze(1) + rc.s_max, min = 0, max = 2 * rc.s_max)
            rel_chain_enc = F.one_hot(d_chain, num_classes = 2 * rc.s_max + 1)
            d_res = torch.clamp(atom_residue_idx_i.unsqueeze(0) - atom_residue_idx_i.unsqueeze(1) + rc.r_max, min = 0, max = 2 * rc.r_max)
            d_res[~same_chain] = 2 * rc.r_max + 1
            rel_res_enc = F.one_hot(d_res, num_classes = 2 * rc.r_max + 2)
            positional_enc_i = torch.cat([rel_res_enc, rel_atom_enc, rel_chain_enc], dim = -1)
            positional_enc_topk_i = gather_pos_enc(ids_topk_i, positional_enc_i)
            positional_enc_topk[start_a: end_a, :, :] = positional_enc_topk_i

        # fill ids_topk and positional_enc_topk for entire batch with current example
        ids_topk[start_a: end_a, :] = ids_topk_i + start_a + 1

    return q, ids_topk, unmasked_packed_X, num_atoms_per_residue, positional_enc_topk, tot_num_atoms

def aggregate(
    h_A,
    num_atoms_per_residue,
    hidden_dim,
    aggregation_mode
):
    # Calculate the total number of residues
    num_residues = num_atoms_per_residue.size(0)

    # Generate residue indices that map each atom to its corresponding residue
    residue_indices = (
        torch.arange(num_residues, device=h_A.device)
        .repeat_interleave(num_atoms_per_residue)
        .unsqueeze(-1)
        .expand_as(h_A)
    )

    # Initialize the aggregated residue tensor
    h_R = torch.zeros(num_residues, hidden_dim, dtype=h_A.dtype, device=h_A.device)

    # Aggregate atom features to residue-level using scatter_reduce
    h_R.scatter_reduce(src=h_A, dim=0, index=residue_indices, reduce=aggregation_mode)

    return h_R

### GVP AND GCP UTILS

def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)


def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    v_expand = torch.unsqueeze(values, -1)
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
            torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )

def orientations(X):
    forward = normalize(X[:, 1:] - X[:, :-1])
    backward = normalize(X[:, :-1] - X[:, 1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def sidechains(X):
    n, origin, c = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    c, n = normalize(c - origin), normalize(n - origin)
    bisector = normalize(c + n)
    perp = normalize(torch.cross(c, n, dim=-1))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec

def dihedrals(X, eps=1e-7):
    X = torch.flatten(X[:, :, :3], 1, 2)
    bsz = X.shape[0]
    dX = X[:, 1:] - X[:, :-1]
    U = normalize(dX, dim=-1)
    u_2 = U[:, :-2]
    u_1 = U[:, 1:-1]
    u_0 = U[:, 2:]

    # Backbone normals
    n_2 = normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [bsz, -1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], -1)
    return D_features

def positional_embeddings(edge_index,
                           num_embeddings=None,
                           num_positional_embeddings=16,
                           period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    num_embeddings = num_embeddings or num_positional_embeddings
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32,
            device=edge_index.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

def dist(X, E_idx, padding_mask):
    """ Pairwise euclidean distances """
    residue_mask = ~padding_mask
    residue_mask_2D = torch.unsqueeze(residue_mask,1) * torch.unsqueeze(residue_mask,2)
    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    D = norm(dX, dim=-1)

    # sorting preference: first those with coords,then the
    # residues that came from padding are last
    D_adjust = nan_to_num(D) + (~residue_mask_2D) * (1e10)
    D_neighbors = torch.gather(D_adjust, 2, E_idx)

    residue_mask_neighbors = (D_neighbors < 5e9)
    return D_neighbors, residue_mask_neighbors

def rotate(v, R):
    """
    Rotates a vector by a rotation matrix.

    Args:
        v: 3D vector, tensor of shape (length x batch_size x channels x 3)
        R: rotation matrix, tensor of shape (length x batch_size x 3 x 3)

    Returns:
        Rotated version of v by rotation matrix R.
    """
    R = R.unsqueeze(-3)
    v = v.unsqueeze(-1)
    return torch.sum(v * R, dim=-2)


def get_rotation_frames(coords):
    """
    Returns a local rotation frame defined by N, CA, C positions.

    Args:
        coords: coordinates, tensor of shape (batch_size x length x 3 x 3)
        where the third dimension is in order of N, CA, C

    Returns:
        Local relative rotation frames in shape (batch_size x length x 3 x 3)
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)
    return R


#### PROTEIN-MPNN UTILS

# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn



def backbone_coords_to_frames(x_bb: TensorType["... 4 3", float],
                              atom_mask: TensorType["... 4", float],
                              eps=1e-8):
    """
    Convert backbone coordinates to local frames (rotation + translation) for each residue.
    """
    base_atom_names = ["C", "CA", "N"]
    rigid_group_base_atom37_idx = torch.tensor([rc.bb_atom_order[atom] for atom in base_atom_names])
    base_atom_pos = x_bb[..., rigid_group_base_atom37_idx, :]
    gt_frames = Rigid.from_3_points(
            p_neg_x_axis=base_atom_pos[..., 0, :],
            origin=base_atom_pos[..., 1, :],
            p_xy_plane=base_atom_pos[..., 2, :],
            eps=eps,
    )
    gt_exists = torch.min(atom_mask[..., rigid_group_base_atom37_idx], dim=-1)[0]

    rots = torch.eye(3, dtype=x_bb.dtype, device=x_bb.device)
    rots = torch.tile(rots, (*x_bb.shape[:-2], 1, 1))
    rots[..., 0, 0] = -1
    rots[..., 2, 2] = -1

    rots = Rotation(rot_mats=rots)
    gt_frames = gt_frames.compose(Rigid(rots, None))

    gt_frames_tensor = gt_frames.to_tensor_4x4()

    return gt_frames_tensor, gt_exists


def transform_sidechain_frame(x_scn: TensorType["b n 33 3", float],
                              x_bb: TensorType["b n 4 3", float],
                              atom_mask_scn: TensorType["b n 33", float],
                              atom_mask_bb: TensorType["b n 4", float],
                              to_local: bool) -> Tuple[
                                  TensorType["b n 33 3", float],
                                  TensorType["b n", float]
                              ]:
    """
    Transform sidechain coordinates based on the backbone frame.
    If to_local, transform from global to local frame. Otherwise, transform from local to global frame.
    """
    bb_frames, bb_frames_exists = backbone_coords_to_frames(x_bb, atom_mask_bb)
    T = Rigid.from_tensor_4x4(bb_frames[..., None, :, :])

    if to_local:
        # Transform from global to local frame, ghost atom value is at 0
        x_scn = T.invert_apply(x_scn)
        ghost_atom_value = 0
    else:
        # Transform from local to global frame, ghost atom value is at CA
        x_scn = T.apply(x_scn)
        ca_idx = rc.bb_atom_order["CA"]
        ghost_atom_value = x_bb[..., ca_idx:ca_idx + 1, :]

    x_scn = torch.where(atom_mask_scn[..., None].bool(), x_scn, ghost_atom_value)  # "zero out" ghost atoms and missing atoms
    x_scn = torch.where(bb_frames_exists[..., None, None].bool(), x_scn, ghost_atom_value)  # "zero out" sidechain atoms where backbone frame does not exist

    return x_scn, bb_frames_exists
