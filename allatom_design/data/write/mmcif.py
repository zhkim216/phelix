import io
import warnings
from collections.abc import Iterator
from typing import Optional

import ihm
import modelcif
import torch
from modelcif import Assembly, AsymUnit, Entity, System, dumper
from modelcif.model import AbInitioModel, Atom, ModelGroup
from rdkit import Chem
from torch import Tensor
from torchtyping import TensorType

from allatom_design.data import const
from allatom_design.data.data import to
from allatom_design.data.types import Structure

# Ignore warnings about empty entities in mmCIF files
warnings.filterwarnings(
    "ignore",
    message=r"At least one empty Entity.*",
    category=UserWarning,
    module=r"ihm\.dumper"
)


def to_mmcif(structure: Structure, plddts: Optional[Tensor] = None) -> str:  # noqa: C901, PLR0915, PLR0912
    """Write a structure into an MMCIF file.

    Parameters
    ----------
    structure : Structure
        The input structure

    Returns
    -------
    str
        the output MMCIF file

    """
    system = System()

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    # Map entities to chain_ids
    entity_to_chains = {}
    entity_to_moltype = {}

    for chain in structure.chains:
        entity_id = chain["entity_id"]
        mol_type = chain["mol_type"]
        entity_to_chains.setdefault(entity_id, []).append(chain)
        entity_to_moltype[entity_id] = mol_type

    # Map entities to sequences
    sequences = {}
    for entity in entity_to_chains:
        # Get the first chain
        chain = entity_to_chains[entity][0]

        # Get the sequence
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        residues = structure.residues[res_start:res_end]
        sequence = [str(res["name"]) for res in residues]
        sequences[entity] = sequence

    # Create entity objects
    lig_entity = None
    entities_map = {}
    for entity, sequence in sequences.items():
        mol_type = entity_to_moltype[entity]

        if mol_type == const.chain_type_ids["PROTEIN"]:
            alphabet = ihm.LPeptideAlphabet()
            chem_comp = lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")  # noqa: E731
        elif mol_type == const.chain_type_ids["DNA"]:
            alphabet = ihm.DNAAlphabet()
            chem_comp = lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif mol_type == const.chain_type_ids["RNA"]:
            alphabet = ihm.RNAAlphabet()
            chem_comp = lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif len(sequence) > 1:
            alphabet = {}
            chem_comp = lambda x: ihm.SaccharideChemComp(id=x)  # noqa: E731
        else:
            alphabet = {}
            chem_comp = lambda x: ihm.NonPolymerChemComp(id=x)  # noqa: E731

        # Handle smiles
        if len(sequence) == 1 and (sequence[0] == "LIG"):
            if lig_entity is None:
                seq = [chem_comp(sequence[0])]
                lig_entity = Entity(seq)
            model_e = lig_entity
        else:
            seq = [
                alphabet[item] if item in alphabet else chem_comp(item)
                for item in sequence
            ]
            model_e = Entity(seq)

        for chain in entity_to_chains[entity]:
            chain_idx = chain["asym_id"]
            entities_map[chain_idx] = model_e

    # We don't assume that symmetry is perfect, so we dump everything
    # into the asymmetric unit, and produce just a single assembly
    asym_unit_map = {}
    for chain in structure.chains:
        # Define the model assembly
        chain_idx = chain["asym_id"]
        label_asym_id = chr(chain_idx + 65)  # rename to A,B,C, etc.
        asym = AsymUnit(
            entities_map[chain_idx],
            details="Model subunit %s" % label_asym_id,
            id=label_asym_id,
        )
        asym_unit_map[chain_idx] = asym
    modeled_assembly = Assembly(asym_unit_map.values(), name="Modeled assembly")

    class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Predicted lddt"

    class _MyModel(AbInitioModel):
        def get_atoms(self) -> Iterator[Atom]:
            # Add all atom sites.
            res_num = 0
            for chain in structure.chains:
                # We rename the chains in alphabetical order
                het = chain["mol_type"] == const.chain_type_ids["NONPOLYMER"]
                chain_idx = chain["asym_id"]
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]

                residues = structure.residues[res_start:res_end]
                for residue in residues:
                    atom_start = residue["atom_idx"]
                    atom_end = residue["atom_idx"] + residue["atom_num"]
                    atoms = structure.atoms[atom_start:atom_end]
                    atom_coords = atoms["coords"]
                    for i, atom in enumerate(atoms):
                        # This should not happen on predictions, but just in case.
                        if not atom["is_present"]:
                            continue

                        name = atom["name"]
                        name = [chr(c + 32) for c in name if c != 0]
                        name = "".join(name)
                        element = periodic_table.GetElementSymbol(
                            atom["element"].item()
                        )
                        element = element.upper()
                        residue_index = residue["res_idx"] + 1
                        pos = atom_coords[i]
                        biso = (
                            100.00
                            if plddts is None
                            else round(plddts[res_num].item() * 100, 2)
                        )
                        yield Atom(
                            asym_unit=asym_unit_map[chain_idx],
                            type_symbol=element,
                            seq_id=residue_index,
                            atom_id=name,
                            x=f"{pos[0]:.5f}",
                            y=f"{pos[1]:.5f}",
                            z=f"{pos[2]:.5f}",
                            het=het,
                            biso=biso,
                            occupancy=1,
                        )

                    res_num += 1

        def add_plddt(self, plddts):
            res_num = 0
            for chain in structure.chains:
                chain_idx = chain["asym_id"]
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]
                residues = structure.residues[res_start:res_end]
                # We rename the chains in alphabetical order
                for residue in residues:
                    residue_idx = residue["res_idx"] + 1
                    self.qa_metrics.append(
                        _LocalPLDDT(
                            asym_unit_map[chain_idx].residue(residue_idx),
                            round(plddts[res_num].item() * 100, 2),
                        )
                    )
                    res_num += 1

    # Add the model and modeling protocol to the file and write them out:
    model = _MyModel(assembly=modeled_assembly, name="Model")
    if plddts is not None:
        model.add_plddt(plddts)

    model_group = ModelGroup([model], name="All models")
    system.model_groups.append(model_group)

    fh = io.StringIO()
    dumper.write(fh, [system])
    return fh.getvalue()



def write_structure_to_mmcif(structure: Structure,
                   filename: str,
                   plddts: TensorType["n"] | None = None):
    """
    Small wrapper around to_mmcif that writes to a file.
    """
    with open(filename, "w") as f:
        f.write(to_mmcif(structure, plddts))


def write_batched_structures_to_mmcif(structures: list[Structure],
                           filenames: list[str],
                           plddts: TensorType["b n ..."] | None = None):
    """
    Write a list of structures to a list of files.
    """
    for i, (structure_i, filename_i) in enumerate(zip(structures, filenames)):
        if plddts is not None:
            plddts_i = plddts[i]
        else:
            plddts_i = None
        write_structure_to_mmcif(structure_i, filename_i, plddts_i)


def write_feats_to_mmcif(feats: dict[str, TensorType["b n ..."]],
                         filenames: list[str]) -> None:
    """
    Convert a batched dictionary of features to a list of files.

    By default, we use the sequence in the features to determine label_seq_id.
    Since features may be cropped (e.g. for a motif), we also save the original residue indices in the auth_seq_id field.
    """
    periodic_table = Chem.GetPeriodicTable()  # for element mapping

    # Unbatch feats into a list of dicts
    feats_list = []
    for i in range(feats["residue_index"].shape[0]):
        feats_list.append({k: v[i] for k, v in to(feats, "cpu").items()})

    for i, feats_i in enumerate(feats_list):
        system = System()

        # Map each chain to its moltype
        chain_to_moltype = {}  # dict[int, int]

        for chain_id in feats_i["asym_id"].unique():
            chain_id = chain_id.item()
            mol_type = feats_i["mol_type"][feats_i["asym_id"] == chain_id].unique().tolist()
            assert len(mol_type) == 1, f"Expected exactly one unique mol_type for chain {chain_id}, got {len(mol_type)}"
            chain_to_moltype[chain_id] = mol_type[0]

        # Map chains to sequences
        sequences = {}
        for chain_id in chain_to_moltype:
            # Get the unpadded sequence
            res_type = feats_i["res_type"].argmax(dim=-1)
            res_type = res_type[feats_i["token_pad_mask"].bool()].tolist()
            sequence = [const.tokens[res_type[ri]] for ri in range(len(res_type))]
            sequences[chain_id] = sequence

        # Create entity objects, assuming each chain is a separate entity
        lig_entity = None
        entities_map = {}
        for chain_id, sequence in sequences.items():
            mol_type = chain_to_moltype[chain_id]

            if mol_type == const.chain_type_ids["PROTEIN"]:
                alphabet = ihm.LPeptideAlphabet()
                chem_comp = lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")  # noqa: E731
            elif mol_type == const.chain_type_ids["DNA"]:
                alphabet = ihm.DNAAlphabet()
                chem_comp = lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
            elif mol_type == const.chain_type_ids["RNA"]:
                alphabet = ihm.RNAAlphabet()
                chem_comp = lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
            elif len(sequence) > 1:
                alphabet = {}
                chem_comp = lambda x: ihm.SaccharideChemComp(id=x)  # noqa: E731
            else:
                alphabet = {}
                chem_comp = lambda x: ihm.NonPolymerChemComp(id=x)  # noqa: E731

            # Handle smiles
            if len(sequence) == 1 and (sequence[0] == "LIG"):
                if lig_entity is None:
                    seq = [chem_comp(sequence[0])]
                    lig_entity = Entity(seq)
                model_e = lig_entity
            else:
                seq = [
                    alphabet[item] if item in alphabet else chem_comp(item)
                    for item in sequence
                ]
                model_e = Entity(seq)

            entities_map[chain_id] = model_e  # each chain is an entity

        # We don't assume that symmetry is perfect, so we dump everything
        # into the asymmetric unit, and produce just a single assembly
        asym_unit_map = {}
        for chain_id in chain_to_moltype:
            # Map from label_seq_id to auth_seq_id
            auth_seq_ids = feats_i["residue_index"][feats_i["token_pad_mask"].bool()].tolist()  # treat input residue indices as auth_seq_ids
            auth_seq_id_map = {label_seq_id + 1: auth_seq_id for label_seq_id, auth_seq_id in enumerate(auth_seq_ids)}

            # Set chain_tag to A,B,C, etc., 0 indexed
            chain_tag = chr(chain_id + 65)
            asym = AsymUnit(entities_map[chain_id], auth_seq_id_map=auth_seq_id_map, id=chain_tag)
            asym_unit_map[chain_id] = asym

        modeled_assembly = Assembly(asym_unit_map.values(), name="Modeled assembly")

        class _MyModel(AbInitioModel):
            def get_atoms(self) -> Iterator[Atom]:
                # Add all atom sites.
                for chain_id, mol_type in chain_to_moltype.items():
                    het = mol_type == const.chain_type_ids["NONPOLYMER"]

                    # Get label_seq_id for each atom
                    label_seq_ids = torch.arange(1, len(feats_i["residue_index"]) + 1)
                    label_seq_id_atomwise = (feats_i["atom_to_token"].float() @ label_seq_ids.float()).long().tolist()

                    # Convert from one-hot to index
                    atom_names = feats_i["ref_atom_name_chars"].argmax(dim=-1)
                    atom_elements = feats_i["ref_element"].argmax(dim=-1)

                    for ai in range(feats_i["coords"].shape[0]):
                        if not feats_i["atom_pad_mask"][ai] or not feats_i["atom_resolved_mask"][ai]:
                            continue
                        name = atom_names[ai]
                        name = [chr(c + 32) for c in name if c != 0]
                        name = "".join(name)

                        element = periodic_table.GetElementSymbol(atom_elements[ai].item())
                        element = element.upper()

                        pos = feats_i["coords"][ai].tolist()
                        biso = 100.00

                        yield Atom(
                            asym_unit=asym_unit_map[chain_id],
                            type_symbol=element,
                            seq_id=label_seq_id_atomwise[ai],
                            atom_id=name,
                            x=f"{pos[0]:.5f}",
                            y=f"{pos[1]:.5f}",
                            z=f"{pos[2]:.5f}",
                            het=het,
                            biso=biso,
                            occupancy=1)

        # Add the model and modeling protocol to the file and write them out:
        model = _MyModel(assembly=modeled_assembly, name="Model")
        model_group = ModelGroup([model], name="All models")
        system.model_groups.append(model_group)

        fh = io.StringIO()
        dumper.write(fh, [system])
        mmcif_str = fh.getvalue()
        with open(filenames[i], "w") as f:
            f.write(mmcif_str)
