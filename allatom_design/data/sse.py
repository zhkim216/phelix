# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module allows estimation of secondary structure elements in protein
structures.

Adapted from https://github.com/biotite-dev/biotite/blob/v1.3.0/src/biotite/structure/sse.py
"""
import warnings

import numpy as np
from biotite.structure.geometry import angle, dihedral, distance
from biotite.structure.sse import _mask_regions_with_contacts

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")


_r_helix = (np.deg2rad(89 - 12), np.deg2rad(89 + 12))
_a_helix = (np.deg2rad(50 - 20), np.deg2rad(50 + 20))
_d2_helix = ((5.5 - 0.5), (5.5 + 0.5))  # Not used in the algorithm description
_d3_helix = ((5.3 - 0.5), (5.3 + 0.5))
_d4_helix = ((6.4 - 0.6), (6.4 + 0.6))

_r_strand = (np.deg2rad(124 - 14), np.deg2rad(124 + 14))
_a_strand = (np.deg2rad(-180), np.deg2rad(-125), np.deg2rad(145), np.deg2rad(180))
_d2_strand = ((6.7 - 0.6), (6.7 + 0.6))
_d3_strand = ((9.9 - 0.9), (9.9 + 0.9))
_d4_strand = ((12.4 - 1.1), (12.4 + 1.1))


def annotate_sse(ca_coord: np.ndarray, residue_index: np.ndarray, return_as_3state: bool = False) -> np.ndarray:
    """
    residue_index allows for handling non-contiguous residues by adding virtual residues at discontinuities.

    Returns:
        np.ndarray: SSE annotations, 0: unknown, 1: helix, 2: strand, 3: coil
    """
    residue_starts = np.arange(len(ca_coord))

    if len(ca_coord) <= 5:
        # The number of atoms is too small #
        # to measure the distances/angles
        # -> Return an SSE array where each amino acid is 'coil'
        sse = np.full(len(ca_coord), "L", dtype="U1")
        # Residues where coord are NaN do not belong to amino acids
        # (or at least they have no CA)
        sse[np.isnan(ca_coord).any(axis=-1)] = "M"
        if return_as_3state:
            return sse
        return sse_letters_to_numbers(sse)

    # Add virtual residues w/o CA coord at chain discontinuity indices
    # This ensures that such discontinuities are recognized for the
    # purpose of geometric measurements
    # -> the distances/angles spanning discontinuities are NaN
    discont_indices = check_resid_continuity(residue_index)
    discont_res_indices = np.searchsorted(residue_starts, discont_indices, "right") - 1
    ca_coord = np.insert(
        ca_coord,
        discont_res_indices,
        np.full((len(discont_res_indices), 3), np.nan),
        axis=0,
    )
    # Later the SSE for virtual residues are removed again
    # via this mask
    no_virtual_mask = np.ones(len(residue_starts), dtype=bool)
    no_virtual_mask = np.insert(no_virtual_mask, discont_res_indices, False)

    length = len(ca_coord)

    # The distances and angles are not defined for the entire interval,
    # therefore the indices do not have the full range
    # Values that are not defined are NaN
    d2i = np.full(length, np.nan)
    d3i = np.full(length, np.nan)
    d4i = np.full(length, np.nan)
    ri = np.full(length, np.nan)
    ai = np.full(length, np.nan)

    d2i[1 : length - 1] = distance(ca_coord[0 : length - 2], ca_coord[2:length])
    d3i[1 : length - 2] = distance(ca_coord[0 : length - 3], ca_coord[3:length])
    d4i[1 : length - 3] = distance(ca_coord[0 : length - 4], ca_coord[4:length])
    ri[1 : length - 1] = angle(
        ca_coord[0 : length - 2], ca_coord[1 : length - 1], ca_coord[2:length]
    )
    ai[1 : length - 2] = dihedral(
        ca_coord[0 : length - 3],
        ca_coord[1 : length - 2],
        ca_coord[2 : length - 1],
        ca_coord[3 : length - 0],
    )

    # Find CA that meet criteria for potential helices and strands
    relaxed_helix = ((d3i >= _d3_helix[0]) & (d3i <= _d3_helix[1])) | (
        (ri >= _r_helix[0]) & (ri <= _r_helix[1])
    )
    strict_helix = (
        (d3i >= _d3_helix[0])
        & (d3i <= _d3_helix[1])
        & (d4i >= _d4_helix[0])
        & (d4i <= _d4_helix[1])
    ) | (
        (ri >= _r_helix[0])
        & (ri <= _r_helix[1])
        & (ai >= _a_helix[0])
        & (ai <= _a_helix[1])
    )

    relaxed_strand = (d3i >= _d3_strand[0]) & (d3i <= _d3_strand[1])
    strict_strand = (
        (d2i >= _d2_strand[0])
        & (d2i <= _d2_strand[1])
        & (d3i >= _d3_strand[0])
        & (d3i <= _d3_strand[1])
        & (d4i >= _d4_strand[0])
        & (d4i <= _d4_strand[1])
    ) | (
        (ri >= _r_strand[0])
        & (ri <= _r_strand[1])
        & (
            # Account for periodic boundary of dihedral angle
            ((ai >= _a_strand[0]) & (ai <= _a_strand[1]))
            | ((ai >= _a_strand[2]) & (ai <= _a_strand[3]))
        )
    )

    helix_mask = _mask_consecutive(strict_helix, 5)
    helix_mask = _extend_region(helix_mask, relaxed_helix)

    strand_mask = _mask_consecutive(strict_strand, 4)
    short_strand_mask = _mask_regions_with_contacts(
        ca_coord,
        _mask_consecutive(strict_strand, 3),
        min_contacts=5,
        min_distance=4.2,
        max_distance=5.2,
    )
    strand_mask = _extend_region(strand_mask | short_strand_mask, relaxed_strand)

    sse = np.full(length, "L", dtype="U1")
    sse[helix_mask] = "H"
    sse[strand_mask] = "E"
    # Residues where coord are NaN do not belong to amino acids
    # (or at least they have no CA)
    sse[np.isnan(ca_coord).any(axis=-1)] = "M"
    # Remove SSE for virtual atoms and return
    sse = sse[no_virtual_mask]
    if return_as_3state:
        return sse
    return sse_letters_to_numbers(sse)  # map to 0: unknown, 1: helix, 2: strand, 3: coil


def _mask_consecutive(mask, number):
    """
    Find all regions in a mask with `number` consecutive ``True``
    values.
    Return a mask that is ``True`` for all indices in such a region and
    ``False`` otherwise.
    """
    # An element is in a consecutive region,
    # if it and the following `number-1` elements are True
    # The elements `mask[-(number-1):]` cannot have the sufficient count
    # by this definition, as they are at the end of the array
    counts = np.zeros(len(mask) - (number - 1), dtype=int)
    for i in range(number):
        counts[mask[i : i + len(counts)]] += 1
    consecutive_seed = counts == number

    # Not only that element, but also the
    # following `number-1` elements are in a consecutive region
    consecutive_mask = np.zeros(len(mask), dtype=bool)
    for i in range(number):
        consecutive_mask[i : i + len(consecutive_seed)] |= consecutive_seed

    return consecutive_mask


def _extend_region(base_condition_mask, extension_condition_mask):
    """
    Extend a ``True`` region in `base_condition_mask` by at maximum of
    one element at each side, if such element fulfills
    `extension_condition_mask.`
    """
    # This mask always marks the start
    # of either a 'True' or 'False' region
    # Prepend absent region to the start to capture the event,
    # that the first element is already the start of a region
    region_change_mask = np.diff(np.append([False], base_condition_mask))

    # These masks point to the first `False` element
    # left and right of a 'True' region
    # The left end is the element before the first element of a 'True' region
    left_end_mask = region_change_mask & base_condition_mask
    # Therefore the mask needs to be shifted to the left
    left_end_mask = np.append(left_end_mask[1:], [False])
    # The right end is first element of a 'False' region
    right_end_mask = region_change_mask & ~base_condition_mask

    # The 'base_condition_mask' gets additional 'True' elements
    # at left or right ends, which meet the extension criterion
    return base_condition_mask | (
        (left_end_mask | right_end_mask) & extension_condition_mask
    )


def check_resid_continuity(residue_index: np.ndarray) -> np.ndarray:
    """
    Adapted from https://github.com/biotite-dev/biotite/blob/main/src/biotite/structure/integrity.py#L29
    Check if the array is continuous. Return the indices of the discontinuities.

    Used for checking residue index continuity when assigning SSE.
    """
    diff = np.diff(residue_index)
    discontinuity = np.where(((diff != 0) & (diff != 1)))
    return discontinuity[0] + 1


def sse_letters_to_numbers(sse_letter_arr: np.ndarray) -> np.ndarray:
    """
    Map SSE letters to an array of numbers.
    Returns:
        np.ndarray: SSE annotations, 0: unknown, 1: helix, 2: strand, 3: coil
    """
    sse = np.full(sse_letter_arr.shape, 0, dtype=np.int8)
    sse[sse_letter_arr == 'H'] = 1
    sse[sse_letter_arr == 'E'] = 2
    sse[sse_letter_arr == 'L'] = 3
    return sse
