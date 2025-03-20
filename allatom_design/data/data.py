import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchtyping import TensorType

import allatom_design.data.conditioning_labels as cl
import openfold.data.data_transforms as data_transforms
from allatom_design.data import protein
from allatom_design.data import residue_constants as rc
from openfold.utils.feats import atom14_to_atom37
from openfold.utils.rigid_utils import Rigid, Rotation

FEATURES_LONG = ("residue_index", "chain_index", "aatype", "aatype_scaffold")

def load_feats_from_pdb(pdb, chain_ids_override: str = None, max_conformers: int = 1):
    """
    Load model input features from a PDB file or mmcif file.
    - chain_residx_gap: Gap to add between residue indices in different chains.
    - max_conformers: Handle disordered atoms, max number of altlocs to store. If > 1, returns coords with shape [seqlen, num_atoms, max_conformers, 3]
    """
    feats = {}
    protein_obj, chain_id_mapping = protein.read_pdb(pdb, chain_ids_override=chain_ids_override, max_conformers=max_conformers)
    for k, v in vars(protein_obj).items():
        if isinstance(v, list) and all(isinstance(item, np.ndarray) for item in v):
            # convert list of numpy arrays to a single numpy array first if needed
            v = np.array(v)
        feats[k] = torch.tensor(v, dtype=torch.float32)

    feats["all_atom_positions"] = feats.pop("atom_positions")
    feats["all_atom_mask"] = feats.pop("atom_mask")
    feats["aatype"] = feats["aatype"].long()

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
    feats["interface_residue_mask"] = get_interface_residue_mask(feats['all_atom_positions'], feats['chain_index'])

    # Mapping from chain letter to chain index
    feats["chain_id_mapping"] = chain_id_mapping

    return feats

def get_interface_residue_mask(x, chain_index):
    # Extract C-alpha atoms' positions
    x_ca = x[:, 1, :]

    # Calculate pairwise Euclidean distances between C-alpha atoms
    d_ca = x_ca[None, :, :] - x_ca[:,  None, :]
    d_ca = torch.sqrt(torch.sum(d_ca ** 2, dim=2))

    # Create a mask for residues within the same chain
    same_chain_mask = torch.eq(chain_index[:, None], chain_index[None, :])
    d_ca[same_chain_mask] = np.inf  # Set distances within the same chain to infinity

    # Apply cutoff to get interface residues
    within_cutoff = d_ca < rc.interface_cutoff
    interface_residue_mask = torch.any(within_cutoff, dim=1).to(dtype=torch.bool)
    return interface_residue_mask

def check_valid_interface(x, atom_mask, chain_index):
    num_residues, num_atoms_per_residue, _ = x.shape
    x_flat = x.reshape(-1, 3)
    atom_mask_flat = atom_mask.reshape(-1)

    # Create residue index mapping
    residue_index = torch.arange(num_residues, device=x.device).repeat_interleave(num_atoms_per_residue)

    # Mask to filter valid atoms
    valid_mask = atom_mask_flat.bool()
    x_valid = x_flat[valid_mask]
    residue_index_valid = residue_index[valid_mask]
    chain_index_valid = chain_index[residue_index_valid]

    # Calculate pairwise distances only for valid atoms
    d_valid = torch.cdist(x_valid, x_valid, p=2)

    # Mask out same-chain residues
    same_chain_mask = chain_index_valid[:, None] == chain_index_valid[None, :]
    d_valid[same_chain_mask] = float('inf')

    # Check if any inter-chain distance is below the threshold
    return torch.any(d_valid < 5.01)

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


def make_fixed_size_1d(data: TensorType["n ..."], fixed_size: int, start_idx: int, multimer_crop_mask: TensorType["n"] = None):
    data_len = data.shape[0]
    if data_len > fixed_size:
        if multimer_crop_mask is not None:
            new_data = data[multimer_crop_mask]
        else:
            new_data = data[start_idx : (start_idx + fixed_size)]
    else:
        pad_size = fixed_size - data_len
        extra_shape = data.shape[1:]
        new_data = torch.cat([data, torch.zeros(pad_size, *extra_shape)], 0)
    return new_data


def pad_to_max_len(batch: Dict[str, TensorType["b n ..."]], max_len: int):
    """
    Inverse of trim_to_max_len; pads a batch to a fixed length.
    """
    padded_example = {}
    for k, v in batch.items():
        # features which aren't padded
        if k in ['pdb_key', 'cond_labels_in', 'chain_ids']:
            padded_example[k] = v
        else:
            B, N, *extra_shape = v.shape
            padding = torch.zeros((B, max_len - N, *extra_shape), device=v.device, dtype=v.dtype)
            padded_example[k] = torch.cat([v, padding], dim=1)
    return padded_example


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
    assert a.dim() == 3 and b.dim() == 3, "Input tensors must be 3D (batch, num_atoms, 3)"

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
                               translation_scale=1.0,
                               return_transforms=False,
                               apply_random_augmentation: bool = True
                               ):
    """
    Batched or unbatched.
    Mean center on CA atoms, then apply random rotation and translation.
    Ensures that missing/ghost/padding atoms are set back to 0.

    Inputs:
        - seq_mask: 0 if residue is padding
        - atom_mask: 1 if not ghost and not missing atom, 0 otherwise
    """
    input_dim = coords_in.dim()
    if input_dim == 3:
        # unbatched; add batch dimension
        coords_in = coords_in.unsqueeze(0)
        atom_mask = atom_mask.unsqueeze(0)
        seq_mask = seq_mask.unsqueeze(0)

    X = coords_in[:, :, 1:2]  # [b n 1 3]

    # Center coords
    M = atom_mask[:, :, 1:2] * seq_mask[:, :, None]  # [b n 1]
    M_sum = M.sum(dim=1, keepdim=True)[..., None]  # [b 1 1 1]
    coords_mean = (X * M[..., None]).sum(dim=1, keepdim=True) / M_sum  # [b 1 1 3]
    coords_in = coords_in - coords_mean

    if not apply_random_augmentation:
        # Return centered coordinates without random augmentation
        return coords_in.squeeze(0) if input_dim == 3 else coords_in

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
        seq_mask = seq_mask.unsqueeze(0)
        atom_mask = atom_mask.unsqueeze(0)

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

def atom37_to_atom14(aatype: TensorType["... n", int],
                     all_atom_pos: TensorType["... n 37 3", float],
                     atom37_mask: Optional[TensorType["... n 37", float]] = None):
    """Convert Atom37 positions to Atom14 positions."""
    if atom37_mask is None:
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


def get_scaffolding_inputs(sm: Optional["ScaffoldManager"],
                           example: Dict[str, TensorType["..."]]) -> Tuple[TensorType["n 37 3"],
                                                                           TensorType["n 37"],
                                                                           TensorType["n"],
                                                                           TensorType["n 37 3"]]:
    """
    Given a scaffold manager and example, return the scaffolded inputs.
    Centers both the motif and the original coordinates on the CA of the scaffolding residues.

    If sm is None, returns unconditional generation inputs.
    """
    x_recentered = example["x"]
    if sm is None:
        x_motif = torch.zeros_like(example["x"])
        motif_mask = torch.zeros_like(example["atom_mask"])
        aatype_scaffold = torch.full_like(example["residue_index"], fill_value=rc.restype_order_with_x["X"])
    else:
        sm_outputs = sm(example)
        x_motif = sm_outputs["x_motif"]
        motif_mask = sm_outputs["motif_mask"]
        aatype_scaffold = sm_outputs["aatype_scaffold"]
        x_recentered = sm_outputs["x_recentered"]

    return x_motif, motif_mask, aatype_scaffold, x_recentered


def get_length_from_pdb(pdb_file: str) -> Tuple[str, int]:
    data = load_feats_from_pdb(pdb_file)
    return pdb_file, len(data["aatype"])
