import pandas as pd
from atomworks.enums import ChainTypeInfo, ChainType
from atomworks.constants import METAL_ELEMENTS

### select protein chains >= 30 residues
atomworks_parquet = pd.read_parquet("/home/possu/jinho/datasets/atomworks_251025/iter1/metadata.parquet")

### Proteins
protein_chain_types = ChainTypeInfo.PROTEINS
protein_chain_type_values = [chain_type.value for chain_type in protein_chain_types]

### Nucleic acids
DNA_chain_type_values = [ChainType.DNA.value]
RNA_chain_type_values = [ChainType.RNA.value]
RNA_DNA_hybrid_chain_type_values = [ChainType.DNA_RNA_HYBRID.value]

### Ligands
ligand_chain_types = ChainTypeInfo.NON_POLYMERS
ligand_chain_type_values = [chain_type.value for chain_type in ligand_chain_types]

# add "is_protein" & "is_peptide" column, following the definition in Atomworks
atomworks_parquet["q_pn_unit_is_protein"] = (atomworks_parquet["q_pn_unit_type"].isin(protein_chain_type_values)) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] >= 20)
atomworks_parquet["q_pn_unit_is_peptide"] = (atomworks_parquet["q_pn_unit_type"].isin(protein_chain_type_values)) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] < 20)

# DNA polymer & ligand columns, following the definition in Plinder
atomworks_parquet["q_pn_unit_is_DNA_polymer"] = atomworks_parquet["q_pn_unit_type"].isin(DNA_chain_type_values) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] > 10)
atomworks_parquet["q_pn_unit_is_DNA_ligand"] = atomworks_parquet["q_pn_unit_type"].isin(DNA_chain_type_values) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] <= 10)

# RNA polymer & ligand columns, following the definition in Plinder
atomworks_parquet["q_pn_unit_is_RNA_polymer"] = atomworks_parquet["q_pn_unit_type"].isin(RNA_chain_type_values) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] > 10)
atomworks_parquet["q_pn_unit_is_RNA_ligand"] = atomworks_parquet["q_pn_unit_type"].isin(RNA_chain_type_values) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] <= 10)

# RNA-DNA hybrid polymer & ligand columns, following the definition in Plinder
atomworks_parquet["q_pn_unit_is_RNA_DNA_hybrid_polymer"] = atomworks_parquet["q_pn_unit_type"].isin(RNA_DNA_hybrid_chain_type_values) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] > 10)
atomworks_parquet["q_pn_unit_is_RNA_DNA_hybrid_ligand"] = atomworks_parquet["q_pn_unit_type"].isin(RNA_DNA_hybrid_chain_type_values) & (atomworks_parquet["q_pn_unit_num_resolved_residues"] <= 10)

# Small molecule ligands & small molecule - metal complexes
atomworks_parquet["q_pn_unit_is_small_molecule"] = (atomworks_parquet["q_pn_unit_type"].isin(ligand_chain_type_values)) & (atomworks_parquet["q_pn_unit_is_metal"] == False)


