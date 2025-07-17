"""
Contains various utils from Boltz-1 rcsb.py for parsing mmCIF files.
Adapted and extended for use in allatom_design by Richard Shuai.
"""

import hashlib
import json
import pickle
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import gemmi
import numpy as np
from natsort import natsorted
from redis import Redis

from allatom_design.data.filter.static.filter import StaticFilter
from allatom_design.data.preprocessing.boltz_utils.mmcif import parse_mmcif
from allatom_design.data.types import (ChainInfo, Connection, Input,
                                       InterfaceInfo, Record, Structure,
                                       Target)


def process_structure(
    data: "PDB",
    resource: dict,
    outdir: str,
    filters: list[StaticFilter],
    clusters: dict,
    return_struct_path: bool = False,
    parse_mmcif_kwargs: dict = {},
) -> None | str:
    """Process a target.

    Parameters
    ----------
    item : PDB
        The raw input data.
    resource: Resource
        The shared resource.
    outdir : str
        The output directory.

    """
    outdir = Path(outdir)

    Path(outdir / "structures").mkdir(parents=True, exist_ok=True)
    Path(outdir / "records").mkdir(parents=True, exist_ok=True)

    # Check if we need to process
    struct_path = outdir / "structures" / f"{data.id}.npz"
    record_path = outdir / "records" / f"{data.id}.json"

    if struct_path.exists() and record_path.exists():
        if return_struct_path:
            return str(struct_path)
        return

    try:
        # Parse the target
        target: Target = parse(data, resource, clusters, **parse_mmcif_kwargs)
        structure = target.structure

        # Apply the filters
        mask = structure.mask
        if filters is not None:
            for f in filters:
                filter_mask = f.filter(structure)
                mask = mask & filter_mask
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"Failed to parse {data.id}")
        return

    # Replace chains and interfaces
    chains = []
    for i, chain in enumerate(target.record.chains):
        chains.append(replace(chain, valid=bool(mask[i])))

    interfaces = []
    for interface in target.record.interfaces:
        chain_1 = bool(mask[interface.chain_1])
        chain_2 = bool(mask[interface.chain_2])
        interfaces.append(replace(interface, valid=(chain_1 and chain_2)))

    # Replace structure and record
    structure = replace(structure, mask=mask)
    record = replace(target.record, chains=chains, interfaces=interfaces)
    target = replace(target, structure=structure, record=record)

    # Dump structure
    np.savez_compressed(struct_path, **asdict(structure))

    # Dump record
    with record_path.open("w") as f:
        json.dump(asdict(record), f)

    if return_struct_path:
        return str(struct_path)


def finalize(outdir: Path) -> str:
    """Run post-processing in main thread.

    Parameters
    ----------
    outdir : Path
        The output directory.

    """
    # Group records into a manifest
    records_dir = outdir / "records"

    failed_count = 0
    records = []
    for record in natsorted(records_dir.iterdir()):
        path = record
        try:
            with path.open("r") as f:
                records.append(json.load(f))
        except:  # noqa: E722
            failed_count += 1
            print(f"Failed to parse {record}")  # noqa: T201
    if failed_count > 0:
        print(f"Failed to parse {failed_count} entries.")  # noqa: T201
    else:
        print("All entries parsed successfully.")

    # Save manifest
    outpath = outdir / "manifest_unclustered.json"
    with outpath.open("w") as f:
        json.dump(records, f)
    return str(outpath)


@dataclass(frozen=True, slots=True)
class PDB:
    """A raw MMCIF PDB file."""

    id: str
    path: str


def fetch(mmcif_files: list[str | Path], max_file_size: Optional[int] = None) -> list[PDB]:
    """Fetch the PDB files."""
    data = []
    excluded = 0
    for file in mmcif_files:
        # The clustering file is annotated by pdb_entity id
        file = Path(file)
        pdb_id = str(file.stem).lower()

        # Check file size and skip if too large
        if max_file_size is not None and (file.stat().st_size > max_file_size):
            excluded += 1
            continue

        # Create the target
        target = PDB(id=pdb_id, path=str(file))
        data.append(target)

    print(f"Excluded {excluded} files due to size.")  # noqa: T201
    return data



def parse(data: PDB, resource: dict, clusters: dict) -> Target:
    """Process a structure.

    Parameters
    ----------
    data : PDB
        The raw input data.
    resource: Resource
        The shared ccd resource.

    Returns
    -------
    Target
        The processed data.

    """
    # Get the PDB id
    pdb_id = data.id.lower()

    # Parse structure
    parsed = parse_mmcif(data.path, resource)
    structure = parsed.data
    structure_info = parsed.info

    # Create chain metadata
    chain_info = []
    for i, chain in enumerate(structure.chains):
        key = f"{pdb_id}_{chain['entity_id']}"
        chain_info.append(
            ChainInfo(
                chain_id=i,
                chain_name=chain["name"],
                msa_id=-1,  # FIX
                mol_type=int(chain["mol_type"]),
                cluster_id=clusters.get(key, -1),
                num_residues=int(chain["res_num"]),
            )
        )

    # Get interface metadata
    interface_info = []
    for interface in structure.interfaces:
        chain_1 = int(interface["chain_1"])
        chain_2 = int(interface["chain_2"])
        interface_info.append(
            InterfaceInfo(
                chain_1=chain_1,
                chain_2=chain_2,
            )
        )

    # Create record
    record = Record(
        id=data.id,
        structure=structure_info,
        chains=chain_info,
        interfaces=interface_info,
    )

    return Target(structure=structure, record=record)


def pdb_to_mmcif(pdb_path: str, mmcif_out: Path,
                 assign_label_seq_id: bool,
                 set_seqid_gaps_to_X: bool = False,
                 ) -> None:
    """
    Convert a PDB file to mmCIF format using gemmi.
    """
    if Path(mmcif_out).exists():
        return

    structure = gemmi.read_structure(pdb_path)
    structure.setup_entities()

    if assign_label_seq_id:
        assert not set_seqid_gaps_to_X, "set_gaps_to_X is not supported when assign_label_seq_id is True"

        # automatically assign label_seq_id by aligning the sequence in the model with the sequence in the SEQRES
        structure.assign_label_seq_id()
    else:
        # Set sequence for each entity based on the sequence in the model, since we do not have SEQRES in these files

        # create mapping from subchain id to entity
        entities: dict[str, gemmi.Entity] = {}
        for entity in structure.entities:
            entity: gemmi.Entity
            if entity.entity_type.name == "Water":
                continue
            for subchain_id in entity.subchains:
                entities[subchain_id] = entity

        # set sequence for each entity
        for raw_chain in structure[0].subchains():
            model_sequence = raw_chain.extract_sequence()
            subchain_id = raw_chain.subchain_id()
            if set_seqid_gaps_to_X:
                # assume first residue is the start of the sequence, and where there are gaps, we should set SEQRES to X
                # note we also truncate missing residues at the end, since this is impossible to know without the true SEQRES records
                new_sequence = []
                residues = list(raw_chain)
                residx = residues[0].seqid.num

                for idx, residue in enumerate(residues):
                    curr_num = residue.seqid.num
                    # fill in “X” for every missing seqid up to the next real residue
                    if curr_num > residx:
                        for missing in range(residx, curr_num):
                            new_sequence.append("UNK")
                        residx = curr_num

                    # append this residue’s 3-letter code (in the same order as model_sequence)
                    new_sequence.append(model_sequence[idx])

                    # if this is the last residue at this seqid (i.e. no more insertions here), bump residx
                    is_last_of_num = (idx + 1 == len(residues)) or (residues[idx+1].seqid.num != curr_num)
                    if is_last_of_num:
                        residx += 1
                model_sequence = new_sequence

            entities[subchain_id].full_sequence = model_sequence

            if set_seqid_gaps_to_X:
                structure.assign_label_seq_id()

    # Write mmCIF file
    mmcif_doc = structure.make_mmcif_document()
    mmcif_doc.write_file(str(mmcif_out))


def mmcif_to_pdb(mmcif_path: str, pdb_out: Path, assign_label_seq_id: bool,
                 overwrite: bool = False) -> None:
    """
    Convert a mmCIF file to PDB format using gemmi.
    """
    if Path(pdb_out).exists() and not overwrite:
        return

    structure = gemmi.read_structure(mmcif_path)
    structure.setup_entities()

    if assign_label_seq_id:
        # automatically assign label_seq_id by aligning the sequence in the model with the sequence in the SEQRES
        structure.assign_label_seq_id()
    else:
        # Set sequence for each entity based on the sequence in the model, since we do not have SEQRES in these files
        entities: dict[str, gemmi.Entity] = {}
        for entity in structure.entities:
            entity: gemmi.Entity
            if entity.entity_type.name == "Water":
                continue
            for subchain_id in entity.subchains:
                entities[subchain_id] = entity

        for raw_chain in structure[0].subchains():
            model_sequence = raw_chain.extract_sequence()
            subchain_id = raw_chain.subchain_id()
            entities[subchain_id].full_sequence = model_sequence

    # Write PDB file
    pdb_str = structure.make_pdb_string()
    with open(pdb_out, "w") as f:
        f.write(pdb_str)


class Resource:
    """Lightweight handle to CCD data stored once in Redis."""

    def __init__(self, host: str, port: int) -> None:
        self._redis = Redis(host=host, port=port)

    def get(self, key: str, default = None):
        value = self._redis.get(key)
        return default if value is None else pickle.loads(value)  # noqa: S301

    def __getitem__(self, key: str):
        out = self.get(key)
        if out is None:
            raise KeyError(key)
        return out


def load_input(structure_path: str) -> Input:
    """Load the given input data.

    Parameters
    ----------
    structure_path : str
        The path to the structure file.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    structure = np.load(structure_path)
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    return Input(structure, msa={})  # we don't load in the MSAs


def split_ensemble_cif(ensemble_cif_path: str, out_dir: str) -> list[str]:
    """
    Split an ensemble CIF file into individual CIF files.
    Returns a list of output file paths.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    structure = gemmi.read_structure(ensemble_cif_path)

    out_files = []
    for i, model in enumerate(structure):
        new_structure = gemmi.Structure()
        new_structure.name = f"{structure.name}_model_{i}"
        new_structure.add_model(model.clone())
        new_structure.setup_entities()
        new_structure.assign_label_seq_id()
        out_file = Path(out_dir) / f"{Path(ensemble_cif_path).stem}_model_{i}.cif"
        new_structure.make_mmcif_document().write_file(str(out_file))
        out_files.append(str(out_file))

    return out_files


def hash_sequence(seq: str) -> str:
    """Hash a sequence."""
    return hashlib.sha256(seq.encode()).hexdigest()
