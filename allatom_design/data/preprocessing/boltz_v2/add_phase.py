
import glob
import hashlib
import json
import os
import pickle
import subprocess
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import gemmi
import hydra
import lightning as L
import pandas as pd
import redis
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data import const
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import (
    Resource, load_input)
from allatom_design.data.types import Manifest, Record


@hydra.main(config_path="../../../configs/data/preprocessing/boltz_v2", config_name="add_phase", version_base="1.3.2",)
def main(cfg: DictConfig) -> None:
    """
    Add phase to manifest.json.
    """
    # Set seed
    L.seed_everything(cfg.seed)
    
    # Load in manifest.json and add phase to the manifest
    processed_targets_dir = f"{cfg.pdb_path}/processed_targets"
    manifest = Manifest.load(f"{processed_targets_dir}/manifest.json")
    
    # Load in validation split
    with open(f"{cfg.pdb_path}/splits/validation_ids.txt", "r") as f:
        val_split = {x.lower() for x in f.read().splitlines()}

    key_to_phase = {}
    for record in manifest.records:
        key_to_phase[record.id.lower()] = "train" if record.id.lower() not in val_split else "eval"

    fn = partial(add_phase_to_record, key_to_phase=key_to_phase)
    new_records = [fn(record) for record in tqdm(manifest.records, desc="Adding phase to manifest")]
    new_records = [asdict(r) for r in new_records]

    with open(f"{processed_targets_dir}/manifest.json", "w") as f:
        json.dump(new_records, f)


def add_phase_to_record(record: Record, key_to_phase: dict[str, str]) -> Record:
    """Returns a new record with the phase added."""
    return replace(record, phase=key_to_phase[record.id.lower()])


if __name__ == "__main__":
    main()
