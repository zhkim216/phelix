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
from openfold.utils.rigid_utils import Rigid
import subprocess
from typing import Tuple, Union


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
