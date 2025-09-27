import copy
import io
import warnings
from collections.abc import Iterator
from functools import partial
from typing import Optional

import ihm
import modelcif
import torch
import torch.nn.functional as F
from modelcif import Assembly, AsymUnit, Entity, System, dumper
from modelcif.model import AbInitioModel, Atom, ModelGroup
from rdkit import Chem
from torch import Tensor
from torchtyping import TensorType

from allatom_design.data import const, data
from allatom_design.data.data import to
from allatom_design.utils.feature_utils import unbatch_feats
from allatom_design.data.types import Structure
from collections import defaultdict

# Ignore warnings about empty entities in mmCIF files
warnings.filterwarnings(
    "ignore",
    message=r"At least one empty Entity.*",
    category=UserWarning,
    module=r"ihm\.dumper"
)


def to_mmcif(structure: Structure, plddts: Optional[Tensor] = None,
             keep_auth: bool = False
             ) -> str:  # noqa: C901, PLR0915, PLR0912
    """Write a structure into an MMCIF file.

    Parameters
    ----------
    structure : Structure
        The input structure
    keep_auth : bool
        If True, save auth_seq_id by mapping label_seq_id to auth_seq_id

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

        if keep_auth:
            # Map from label_seq_id to auth_seq_id
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            residues = structure.residues[res_start:res_end]
            auth_seq_ids = residues["auth_seq_id"].tolist()
            pdb_icodes = [icode.strip() for icode in residues["pdb_icode"]]
            paired = list(zip(auth_seq_ids, pdb_icodes))
            label_seq_ids = residues["res_idx"].tolist()
            auth_seq_id_map = {label_seq_id + 1: pair for label_seq_id, pair in zip(label_seq_ids, paired)}
        else:
            auth_seq_id_map = 0

        # Set label_asym_id to A,B,C, etc. 0 indexed
        label_asym_id = chr(chain_idx + 65)
        asym = AsymUnit(
            entities_map[chain_idx],
            details="Model subunit %s" % label_asym_id,
            id=label_asym_id,
            auth_seq_id_map=auth_seq_id_map,
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


######## allatom-design functions ########
def create_assembly_from_feats(feats: dict[str, TensorType["n ..."]],
                               input_struct: Structure | None,
                               keep_auth: bool = False) -> tuple[Assembly, dict[int, AsymUnit]]:
    """
    Create an assembly from a dictionary of Boltz features. Also returns the asym_unit_map mapping from chain_id to asym_unit within the assembly.
    """
    # First, map each sequence to chains to determine unique entities
    seq_to_chains = defaultdict(list)
    seq_to_moltype = {}
    for chain_id in feats["asym_id"].unique().tolist():
        chain_mask = feats["asym_id"] == chain_id
        mol_type = feats["mol_type"][chain_mask].unique().tolist()[0]
        if mol_type != const.chain_type_ids["NONPOLYMER"]:
            # Extract sequence from the features, since we may have redesigned these
            # Get the unpadded sequence for this chain
            res_type = feats["res_type"][chain_mask].argmax(dim=-1)
            res_type = res_type[feats["token_pad_mask"][chain_mask].bool()].tolist()
            sequence = [const.tokens[res_type[ri]] for ri in range(len(res_type))]
        elif mol_type == const.chain_type_ids["NONPOLYMER"]:
            if input_struct is None:
                raise ValueError("input_struct is required for labeling non-polymer chains")

            # construct chain map (map asym_id to index of chain in the input structure)
            chain_map = {c["asym_id"]: i for i, c in enumerate(input_struct.chains)}
            chain_i = input_struct.chains[chain_map[chain_id]]
            res_start = chain_i["res_idx"]
            res_end = chain_i["res_idx"] + chain_i["res_num"]
            sequence = input_struct.residues[res_start:res_end]["name"].tolist()

        # map sequence to chains and moltype
        seq_to_chains[tuple(sequence)].append(chain_id)
        seq_to_moltype[tuple(sequence)] = mol_type

    # Now, map a new entity_id to each sequence
    sequences = {k: list(v) for k, v in zip(range(len(seq_to_chains)), seq_to_chains.keys())}

    # Map entities to chain_ids and moltypes
    entity_to_chains = {}
    entity_to_moltype = {}
    for entity_id, sequence in sequences.items():
        entity_to_chains[entity_id] = seq_to_chains[tuple(sequence)]
        entity_to_moltype[entity_id] = seq_to_moltype[tuple(sequence)]

    # Create entity objects
    lig_entity = None
    entities_map = {}
    for entity_id, sequence in sequences.items():
        mol_type = entity_to_moltype[entity_id]

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

        for chain_id in entity_to_chains[entity_id]:
            entities_map[chain_id] = model_e

    # We don't assume that symmetry is perfect, so we dump everything
    # into the asymmetric unit, and produce just a single assembly
    asym_unit_map = {}
    for chain_id in feats["asym_id"].unique().tolist():
        # Crop feats to this chain
        chain_mask = feats["asym_id"] == chain_id
        chain_feats_i = crop_sd_feats(feats, chain_mask, max_tokens=None, max_atoms=None, max_seqs=None, in_place=False)

        if keep_auth:
            # Map from label_seq_id to auth_seq_id
            auth_seq_ids = chain_feats_i["auth_seq_id"][chain_feats_i["token_pad_mask"].bool()].tolist()
            pdb_icodes = chain_feats_i["pdb_icode"][chain_feats_i["token_pad_mask"].bool()].tolist()
            paired = [(seq_id, chr(icode + 32).strip()) for seq_id, icode in zip(auth_seq_ids, pdb_icodes)]  # pair up auth_seq_id and pdb_icode
            label_seq_ids = chain_feats_i["label_seq_id"][chain_feats_i["token_pad_mask"].bool()]
            _, label_seq_ids = torch.unique(label_seq_ids, return_inverse=True)  # make label_seq_ids contiguous, starting from 0
            label_seq_ids = (label_seq_ids + 1).tolist()  # renumber label seq id to 1-indexed
            auth_seq_id_map = {label_seq_id: pair for label_seq_id, pair in zip(label_seq_ids, paired)}
        else:
            auth_seq_id_map = 0

        # Set chain_tag to A,B,C, etc., 0 indexed
        chain_tag = chr(chain_id + 65)
        asym = AsymUnit(entities_map[chain_id], auth_seq_id_map=auth_seq_id_map, id=chain_tag)
        asym_unit_map[chain_id] = asym

    modeled_assembly = Assembly(asym_unit_map.values(), name="Modeled assembly")

    return modeled_assembly, asym_unit_map


class ModelFromFeats(AbInitioModel):
    def __init__(self,
                 assembly: Assembly,
                 name: str,
                 asym_unit_map: dict[int, AsymUnit],
                 feats: dict[str, TensorType["n ..."]],
                 keep_auth: bool):
        """
        Create a model from a dictionary of features. Used for writing Boltz features to mmCIF files.
        Keeps track of the asym_unit_map to avoid duplicating entities / asym_units when writing multiple models to the same mmCIF file.
        """
        super().__init__(assembly=assembly, name=name)

        self.periodic_table = Chem.GetPeriodicTable()  # for element mapping
        self.asym_unit_map = asym_unit_map
        self.feats = feats
        self.keep_auth = keep_auth


    def get_atoms(self) -> Iterator[Atom]:
        # Add all atom sites.
        for chain_id in self.feats["asym_id"].unique().tolist():
            # First, subset relevant feats to this chain
            chain_mask = self.feats["asym_id"] == chain_id
            chain_feats_i = crop_sd_feats(self.feats, chain_mask, max_tokens=None, max_atoms=None, max_seqs=None, in_place=False)

            # Get het flag
            het = chain_feats_i["mol_type"].unique().tolist()[0] == const.chain_type_ids["NONPOLYMER"]

            # Get label_seq_id for each atom
            label_seq_ids = chain_feats_i["label_seq_id"][chain_feats_i["token_pad_mask"].bool()]
            if self.keep_auth:
                _, label_seq_ids = torch.unique(label_seq_ids, return_inverse=True)  # make label_seq_ids contiguous, starting from 0
            label_seq_ids = (label_seq_ids - label_seq_ids.min() + 1)  # renumber label seq id to 1-indexed
            label_seq_id_atomwise = (chain_feats_i["atom_to_token"].float() @ label_seq_ids.float()).long().tolist()

            # Convert from one-hot to index
            atom_names = chain_feats_i["ref_atom_name_chars"].long().argmax(dim=-1)
            atom_elements = chain_feats_i["ref_element"].long().argmax(dim=-1)

            for ai in range(chain_feats_i["coords"].shape[0]):
                if not chain_feats_i["atom_pad_mask"][ai] or not chain_feats_i["atom_resolved_mask"][ai]:
                    continue
                name = atom_names[ai]
                name = [chr(c + 32) for c in name if c != 0]
                name = "".join(name)

                element = self.periodic_table.GetElementSymbol(atom_elements[ai].item())
                element = element.upper()

                pos = chain_feats_i["coords"][ai].tolist()
                biso = 100.00

                yield Atom(
                    asym_unit=self.asym_unit_map[chain_id],
                    type_symbol=element,
                    seq_id=label_seq_id_atomwise[ai],
                    atom_id=name,
                    x=f"{pos[0]:.5f}",
                    y=f"{pos[1]:.5f}",
                    z=f"{pos[2]:.5f}",
                    het=het,
                    biso=biso,
                    occupancy=1)


def write_feats_to_mmcif(feats: dict[str, TensorType["n ..."]],
                         input_struct: Structure | None,
                         filename: str,
                         keep_auth: bool = False) -> None:
    """
    Write a dictionary of features to a mmCIF file.

    If keep_auth is True, we save the auth_seq_id by mapping label_seq_id to auth_seq_id.
    Otherwise, I think we expect label_seq_id to be contiguous, starting from 0.
    """
    system = System()

    feats = crop_sd_feats(feats, feats["token_pad_mask"].bool(), max_tokens=None, max_atoms=None, max_seqs=None, in_place=False)
    assembly, asym_unit_map = create_assembly_from_feats(feats, input_struct, keep_auth)
    model = ModelFromFeats(assembly=assembly, name="Model", asym_unit_map=asym_unit_map, feats=feats, keep_auth=keep_auth)
    model_group = ModelGroup([model], name="All models")
    system.model_groups.append(model_group)

    fh = io.StringIO()
    dumper.write(fh, [system])
    mmcif_str = fh.getvalue()
    with open(filename, "w") as f:
        f.write(mmcif_str)


def batch_write_feats_to_mmcif(feats: dict[str, TensorType["b n ..."]],
                               input_structs: list[Structure | None] | None,  # needed for ligand sequence info
                               filenames: list[str],
                               keep_auth: bool = False) -> None:
    """
    Convert a batched dictionary of sequence design features to a list of files.
    """
    feats = to(feats, "cpu")

    # Unbatch feats into a list of dicts
    feats_list = unbatch_feats(feats)

    # Handle input_structs
    if input_structs is None:
        input_structs = [None] * len(feats_list)

    for feats_i, input_struct_i, filename_i in zip(feats_list, input_structs, filenames):
        write_feats_to_mmcif(feats_i, input_struct_i, filename_i, keep_auth=keep_auth)


def write_motif_feats_to_mmcif(feats: dict[str, TensorType["b n ..."]], *args, **kwargs):
    """
    Write motif features to a mmCIF file.
    Wrapper around batch_write_feats_to_mmcif that handles the renaming of motif-specific keys.
    """
    feats = copy.deepcopy(feats)
    feats["coords"] = feats.pop("motif_coords")
    feats["atom_resolved_mask"] = feats.pop("motif_atom_mask")
    batch_write_feats_to_mmcif(feats, *args, **kwargs)


def write_diffusion_inputs_to_mmcif(feats: dict[str, TensorType["b n ..."]], filenames: list[str],
                                    full_feats: dict[str, TensorType["n ..."]] | None = None) -> None:
    """
    Write diffusion inputs to a list of files. Creates dummy features to resemble boltz features, then calls batch_write_feats_to_mmcif.

    If full_feats is provided, we use it to keep SEQRES records when saving diffusion inputs.
    """
    feats = to(feats, "cpu")
    boltz_feats = diffusion_inputs_to_boltz_feats(feats)

    if full_feats is not None:
        full_feats = to(full_feats, "cpu")

        # overwrite res_type with diffusion inputs
        batch_indices, dest_indices, src_indices = torch.where((full_feats["token_index"].unsqueeze(-1) == boltz_feats["token_index"].unsqueeze(-2)))
        full_feats["res_type"][batch_indices, dest_indices] = boltz_feats["res_type"][batch_indices, src_indices]

        _, atomwise_token_idx = torch.max(full_feats["atom_to_token"], dim=-1)
        atomwise_token_resolved_mask = full_feats["token_resolved_mask"].gather(dim=-1, index=atomwise_token_idx)
        B = boltz_feats["token_index"].shape[0]
        for bi in range(B):
            # create gather_mask as True for all backbone atoms in original structure, False otherwise
            gather_mask = (torch.isin(atomwise_token_idx[bi], boltz_feats["token_index"]) * full_feats["prot_bb_atom_mask"][bi])
            gather_mask = gather_mask * atomwise_token_resolved_mask[bi]  # mask out atoms where token is not resolved
            gather_mask = gather_mask.bool()

            # overwrite coords and atom_resolved_mask with diffusion inputs
            atom_pad_mask_i = boltz_feats["atom_pad_mask"][bi].bool()
            full_feats["coords"][bi][gather_mask] = boltz_feats["coords"][bi][atom_pad_mask_i]
            full_feats["atom_resolved_mask"][bi][gather_mask] = boltz_feats["atom_resolved_mask"][bi][atom_pad_mask_i].bool()

        boltz_feats = full_feats

    batch_write_feats_to_mmcif(boltz_feats, input_structs=None, filenames=filenames, keep_auth=True)


def write_diffusion_inputs_to_ensemble(feats: dict[str, TensorType["b n ..."]],
                                       filename: str,
                                       align_on_first_model: bool = True,
                                       ) -> None:
    """
    Write diffusion inputs to a single ensemble file. Items in the batch are written to separate models in the ensemble.

    If align_on_first_model is True, we CA-align all models to the first model.
    """
    feats = to(feats, "cpu")
    boltz_feats = diffusion_inputs_to_boltz_feats(feats)

    # Create ensemble
    system = System()
    models = []

    # Align on first model
    if align_on_first_model:
        B = boltz_feats["coords"].shape[0]
        ca_atom_mask = torch.zeros_like(boltz_feats["atom_resolved_mask"][0:1])
        ca_atom_mask[:, 1::4] = True  # N, CA, C, O
        _, (boltz_feats["coords"], _) = data.torch_rmsd_weighted(boltz_feats["coords"],
                                                                 boltz_feats["coords"][0:1].expand(B, -1, -1),
                                                                 ca_atom_mask.expand(B, -1), return_aligned=True)

    feats_list = unbatch_feats(boltz_feats)
    feats_list = [crop_sd_feats(feats_i, feats_i["token_pad_mask"].bool(),
                                max_tokens=None, max_atoms=None, max_seqs=None, in_place=False) for feats_i in feats_list]

    # Create assembly from first model
    assembly, asym_unit_map = create_assembly_from_feats(feats_list[0], input_struct=None, keep_auth=True)

    for mi, feats_i in enumerate(feats_list):
        model = ModelFromFeats(assembly=assembly, name=f"model_{mi}", asym_unit_map=asym_unit_map, feats=feats_i, keep_auth=True)
        model.name = f"model_{mi}"
        models.append(model)

    model_group = ModelGroup(models, name="All models")
    system.model_groups.append(model_group)

    fh = io.StringIO()
    dumper.write(fh, [system])
    mmcif_str = fh.getvalue()
    with open(filename, "w") as f:
        f.write(mmcif_str)


def diffusion_inputs_to_boltz_feats(feats: dict[str, TensorType["b n ..."]]) -> dict[str, TensorType["n ..."]]:
    """
    Convert diffusion inputs to Boltz-like features for saving to mmCIF files.
    """
    # First, create dummy features to resemble boltz features
    is_unbatched = feats["seq_mask"].ndim == 1
    if is_unbatched:
        feats = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}

    B, N_tokens = feats["seq_mask"].shape
    N_atoms = N_tokens * 4  # 4 backbone atoms
    boltz_feats = {}
    boltz_feats["token_index"] = feats["token_index"]
    boltz_feats["res_type"] = F.one_hot(torch.full_like(feats["residue_index"], const.token_ids["GLY"], dtype=torch.long), num_classes=len(const.tokens))
    boltz_feats["coords"] = feats["x"][..., const.prot_bb_atom14_idxs, :].reshape(B, N_atoms, 3)  # B, N*4, 3
    boltz_feats["atom_resolved_mask"] = feats["atom_mask"][..., const.prot_bb_atom14_idxs].reshape(B, N_atoms)
    boltz_feats["token_pad_mask"] = feats["seq_mask"]
    boltz_feats["atom_pad_mask"] = feats["seq_mask"].repeat_interleave(4, dim=1)
    boltz_feats["label_seq_id"] = feats.get("label_seq_id", feats["residue_index"])
    boltz_feats["auth_seq_id"] = feats.get("auth_seq_id", feats["residue_index"])
    boltz_feats["pdb_icode"] = feats.get("pdb_icode", torch.zeros_like(feats["residue_index"]))
    boltz_feats["asym_id"] = feats["chain_index"]
    boltz_feats["entity_id"] = feats["entity_id"]
    boltz_feats["mol_type"] = torch.full_like(feats["residue_index"], const.chain_type_ids["PROTEIN"])

    atom_to_token_1d = torch.arange(N_tokens, device=feats["seq_mask"].device).repeat_interleave(4)
    boltz_feats["atom_to_token"] = F.one_hot(atom_to_token_1d.expand(B, -1), num_classes=N_tokens)

    ref_element_1d = torch.tensor([7, 6, 6, 8], device=feats["seq_mask"].device).repeat(N_tokens)  # N, Ca, C, O
    boltz_feats["ref_element"] = F.one_hot(ref_element_1d.expand(B, -1), num_classes=ref_element_1d.max() + 1)
    ref_atom_name_chars_1d = torch.tensor([[ord("N") - 32, 0, 0, 0],  # N, CA, C, O
                                            [ord("C") - 32, ord("A") - 32, 0, 0],
                                            [ord("C") - 32, 0, 0, 0],
                                            [ord("O") - 32, 0, 0, 0]], device=feats["seq_mask"].device).repeat(N_tokens, 1)
    ref_atom_name_chars = ref_atom_name_chars_1d.unsqueeze(0).expand(B, -1, -1)
    boltz_feats["ref_atom_name_chars"] = F.one_hot(ref_atom_name_chars, num_classes=ref_atom_name_chars.max() + 1)

    if is_unbatched:
        boltz_feats = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in boltz_feats.items()}

    return boltz_feats
