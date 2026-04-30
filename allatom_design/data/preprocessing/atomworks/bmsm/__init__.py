"""BMSM (biologically meaningful small molecule) preprocessing utilities.

Self-contained package: every dependency required to go from a raw atomworks
metadata parquet to a fully BMSM-augmented + ligand-clustered parquet lives
under this directory.

Modules
-------
- ``ccd_filter``       Table A3 plinder non-artifact filter producing the
                       ``passed_ccd_codes_metadata_*.txt`` whitelist (CLI entry).
- ``smiles_cache``     CCD -> SMILES cache helpers (RCSB REST + atomworks
                       fallback, parallel fetch, incremental flush).
- ``heavy_atom_cache`` SMILES-derived per-CCD heavy atom counts cache.
- ``augment_metadata_with_bmsm``  BMSM column augmentation (CLI entry).
- ``cluster_ligands``             Complete-linkage clustering on Morgan/ECFP4
                                  Tanimoto fingerprints (CLI entry).
"""
