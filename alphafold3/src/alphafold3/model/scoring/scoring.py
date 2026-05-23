# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Library of scoring methods of the model outputs."""

from alphafold3.model import protein_data_processing
import jax.numpy as jnp
import numpy as np


Array = jnp.ndarray | np.ndarray


def pseudo_beta_fn(
    aatype: Array,
    dense_atom_positions: Array,
    dense_atom_masks: Array,
    is_ligand: Array | None = None,
    use_jax: bool | None = True,
    mask_template_sidechains: bool = False,
    mask_template_sequence: bool = False,
    template_is_protein: Array | None = None,
    template_is_dna: Array | None = None,
    template_is_rna: Array | None = None,
    template_is_other: Array | None = None,
) -> tuple[Array, Array] | Array:
  """Create pseudo beta atom positions and optionally mask.

  Args:
    aatype: [num_res] amino acid types.
    dense_atom_positions: [num_res, NUM_DENSE, 3] vector of all atom positions.
    dense_atom_masks: [num_res, NUM_DENSE] mask.
    is_ligand: [num_res] flag if something is a ligand.
    use_jax: whether to use jax for the computations.
    mask_template_sidechains: If true, use template CA instead of pseudo-beta
      for protein residues.
    mask_template_sequence: If true, use template CA instead of pseudo-beta for
      protein residues. Sequence masking itself is handled by the template
      embedding module.
    template_is_protein: Template residue protein mask.
    template_is_dna: Template residue DNA mask.
    template_is_rna: Template residue RNA mask.
    template_is_other: Template residue other-chain mask.

  Returns:
    Pseudo beta dense atom positions and the corresponding mask.
  """
  if use_jax:
    xnp = jnp
  else:
    xnp = np

  if is_ligand is None:
    is_ligand = xnp.zeros_like(aatype)

  pseudobeta_index_polymer = xnp.take(
      protein_data_processing.RESTYPE_PSEUDOBETA_INDEX, aatype, axis=0
  ).astype(xnp.int32)

  pseudobeta_index = xnp.where(
      is_ligand,
      xnp.zeros_like(pseudobeta_index_polymer),
      pseudobeta_index_polymer,
  )
  del template_is_dna, template_is_rna, template_is_other
  if (
      mask_template_sidechains or mask_template_sequence
  ) and template_is_protein is not None:
    pseudobeta_index = xnp.where(
        template_is_protein.astype(bool),
        xnp.ones_like(pseudobeta_index),
        pseudobeta_index,
    )

  pseudo_beta = xnp.take_along_axis(
      dense_atom_positions, pseudobeta_index[..., None, None], axis=-2
  )
  pseudo_beta = xnp.squeeze(pseudo_beta, axis=-2)

  pseudo_beta_mask = xnp.take_along_axis(
      dense_atom_masks, pseudobeta_index[..., None], axis=-1
  ).astype(xnp.float32)
  pseudo_beta_mask = xnp.squeeze(pseudo_beta_mask, axis=-1)

  return pseudo_beta, pseudo_beta_mask
