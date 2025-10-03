"""
Egret embedding extraction utilities.

This package provides utilities to extract atom-level descriptors from
Rowan's Egret-1 family (MACE-based) given structure files (PDB/mmCIF/SDF, etc.),
and to persist them to Parquet with NPZ fallback.

Quick start:
  - CLI: `python -m allatom_design.data.embeddings.extract_embeddings --help`
  - API: `from allatom_design.data.embeddings.egret_embedder import EgretEmbedder`

Reference:
  - Egret public repo: https://github.com/rowansci/egret-public
"""

__all__ = [
    "egret_embedder",
]


