#!/usr/bin/env python3
import glob
import shutil
from contextlib import nullcontext
from pathlib import Path

import hydra
import pandas as pd
import torch
from joblib import Parallel, delayed
from natsort import natsorted
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data.data import get_seq_from_res_type
from allatom_design.utils.feature_utils import unbatch_feats
from allatom_design.data.types import Manifest
from allatom_design.eval.eval_utils.eval_setup_utils import process_pdb_files
from allatom_design.eval.eval_utils.seq_des_utils import get_sd_batch


@hydra.main(config_path="../../../configs/data/preprocessing/cfold", config_name="get_cfold_conformers", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Given train and test conformations from Cfold, create a dataset of conformers by merging train and test files.
    """
    # Create dataset directory
    conformer_out_dir = f"{cfg.out_dir}/conformers"
    Path(conformer_out_dir).mkdir(parents=True, exist_ok=True)

    ### Save both the train and test files to a conformer directory ###
    train_files = glob.glob(f"{cfg.cfold_train_confs_dir}/*.pdb")
    mapping_data = []
    for train_file in tqdm(train_files, desc="Processing train files"):
        train_id, test_id = Path(train_file).stem.split("_")
        test_file = f"{cfg.cfold_test_confs_dir}/{test_id}_{train_id}.pdb"

        if Path(test_file).exists():
            # Make conformer directory and save train + test files to it
            train_stem = Path(train_file).stem.lower()
            test_stem = Path(test_file).stem.lower()
            conformer_dir = f"{conformer_out_dir}/{train_stem}"
            Path(conformer_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(train_file, f"{conformer_dir}/{train_stem}.pdb")
            shutil.copy(test_file, f"{conformer_dir}/{test_stem}.pdb")
            mapping_data.append({"pdb_key": train_stem, "conformer_dir": Path(conformer_dir).name})
            mapping_data.append({"pdb_key": test_stem, "conformer_dir": Path(conformer_dir).name})

            # Also save all PDB files to a single directory
            all_pdbs_dir = f"{cfg.out_dir}/pdbs"
            Path(all_pdbs_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(train_file, f"{all_pdbs_dir}/{train_stem}.pdb")
            shutil.copy(test_file, f"{all_pdbs_dir}/{test_stem}.pdb")
        else:
            print(f"Matching test file {test_file} does not exist, skipping...")

    print(f"Found {len(mapping_data) // 2} conformers")

    ### Build manifest ###
    cfold_manifest_df = pd.DataFrame(mapping_data).sort_values(by="conformer_dir").reset_index(drop=True)
    cfold_manifest_df["source_pdb_key"] = cfold_manifest_df["pdb_key"].str.split("_").str[0]

    # Load features into memory
    pdb_files = natsorted(glob.glob(f"{cfg.out_dir}/pdbs/*.pdb"))
    record_id_to_feats = get_feats_from_pdb_files(pdb_files, cfg.data_cfg, cfg.num_workers, cfg.pdb_processing_cfg, cfg.out_dir)

    # Get lengths and add to manifest
    record_id_to_length = {record_id: feats_i["token_pad_mask"].sum().long().item() for record_id, feats_i in record_id_to_feats.items()}  # length includes gaps, which we set to X
    cfold_manifest_df["length"] = cfold_manifest_df["pdb_key"].map(record_id_to_length)
    cfold_manifest_df = cfold_manifest_df[~pd.isna(cfold_manifest_df["length"])]  # filter out PDBs that failed to process

    # Remove source_pdb_keys not in boltz_v2 manifest, since we cannot hold these out easily
    manifest = Manifest.load(Path(cfg.boltz_v2_manifest))
    record_ids = set([r.id for r in manifest.records])
    cfold_manifest_df = cfold_manifest_df[cfold_manifest_df["source_pdb_key"].isin(record_ids)].reset_index(drop=True)

    # Get sequences (with gaps) and add to manifest
    record_id_to_seq = {record_id: get_seq_from_res_type(feats_i["res_type"][feats_i["token_pad_mask"].bool()]) for record_id, feats_i in record_id_to_feats.items()}
    cfold_manifest_df["seq"] = cfold_manifest_df["pdb_key"].map(record_id_to_seq)

    # Ensure all sequences within a conformer_dir are the same; also handles where there are residx alignment issues
    cfold_manifest_df = cfold_manifest_df.groupby("conformer_dir").filter(lambda x: len(x["seq"].unique()) == 1)

    # Of the remaining PDBs, remove those with less than 2 conformers
    cfold_manifest_df = cfold_manifest_df.groupby("conformer_dir").filter(lambda x: len(x) >= 2)

    print(f"Saving manifest to {cfg.out_dir}/manifest.csv")
    cfold_manifest_df.to_csv(f"{cfg.out_dir}/manifest.csv", index=False)

    ### Get pdb_name_lists ###
    pdb_name_lists_dir = f"{cfg.out_dir}/pdb_name_lists"
    Path(pdb_name_lists_dir).mkdir(parents=True, exist_ok=True)

    # all conformer dirs
    conformer_df = cfold_manifest_df.drop_duplicates(subset=["conformer_dir"], keep="first")
    print(f"Writing {len(conformer_df)} conformer dirs to {pdb_name_lists_dir}/conformer_dirs.txt")
    conformer_df["conformer_dir"].to_csv(f"{pdb_name_lists_dir}/conformer_dirs.txt", index=False, header=False)

    # 32 <= length <= 512
    conformer_df_L32_512 = conformer_df[(32 <= conformer_df["length"]) & (conformer_df["length"] <= 512)]
    print(f"Writing {len(conformer_df_L32_512)} conformer dirs with 32 <= length <= 512 to {pdb_name_lists_dir}/conformer_dirs_L32_512.txt")
    conformer_df_L32_512["conformer_dir"].to_csv(f"{pdb_name_lists_dir}/conformer_dirs_L32_512.txt", index=False, header=False)

    # 32 <= length <= 256
    conformer_df_L32_256 = conformer_df[(32 <= conformer_df["length"]) & (conformer_df["length"] <= 256)]
    print(f"Writing {len(conformer_df_L32_256)} conformer dirs with 32 <= length <= 256 to {pdb_name_lists_dir}/conformer_dirs_L32_256.txt")
    conformer_df_L32_256["conformer_dir"].to_csv(f"{pdb_name_lists_dir}/conformer_dirs_L32_256.txt", index=False, header=False)

    ### Get holdout pdb keys ###
    holdout_pdb_keys_dir = f"{cfg.out_dir}/holdout_pdb_keys"
    Path(holdout_pdb_keys_dir).mkdir(parents=True, exist_ok=True)

    # all holdout pdb keys
    out_file = f"{holdout_pdb_keys_dir}/holdout_pdb_keys.txt"
    holdout_keys = cfold_manifest_df["source_pdb_key"].tolist()
    print(f"Writing {len(holdout_keys)} holdout record IDs to {out_file}")
    with open(out_file, "w") as f:
        for key in holdout_keys:
            f.write(f"{key}\n")

    # 32 <= length <= 512
    holdout_keys_L32_512 = cfold_manifest_df[(32 <= cfold_manifest_df["length"]) & (cfold_manifest_df["length"] <= 512)]["source_pdb_key"].tolist()
    out_file = f"{holdout_pdb_keys_dir}/holdout_pdb_keys_L32_512.txt"
    print(f"Writing {len(holdout_keys_L32_512)} holdout record IDs with 32 <= length <= 512 to {out_file}")
    with open(out_file, "w") as f:
        for key in holdout_keys_L32_512:
            f.write(f"{key}\n")

    # 32 <= length <= 256
    holdout_keys_L32_256 = cfold_manifest_df[(32 <= cfold_manifest_df["length"]) & (cfold_manifest_df["length"] <= 256)]["source_pdb_key"].tolist()
    out_file = f"{holdout_pdb_keys_dir}/holdout_pdb_keys_L32_256.txt"
    print(f"Writing {len(holdout_keys_L32_256)} holdout record IDs with 32 <= length <= 256 to {out_file}")
    with open(out_file, "w") as f:
        for key in holdout_keys_L32_256:
            f.write(f"{key}\n")


def get_feats_from_pdb_files(pdb_files: list[str],
                             data_cfg: DictConfig,
                             num_workers: int,
                             pdb_processing_cfg: DictConfig,
                             out_dir: str) -> dict[str, torch.Tensor]:
    processed_struct_dir = f"{out_dir}/processed_structures"
    processed_struct_files = process_pdb_files(pdb_files, processed_struct_dir=processed_struct_dir, **pdb_processing_cfg)
    data_cfg = hydra.utils.instantiate(data_cfg)
    parallel_context = Parallel(n_jobs=num_workers) if num_workers > 1 else nullcontext()  # for loading PDBs in parallel

    record_id_to_feats = {}  # store features in memory
    with parallel_context:
        B = 32
        for i in tqdm(range(0, len(processed_struct_files), B), desc="Loading PDB features into memory"):
            batch_struct_files = processed_struct_files[i:i+B]
            batch_struct_files = [x for x in batch_struct_files if x is not None]  # filter out PDBs that failed to process
            batch, input_structs = get_sd_batch(batch_struct_files, device="cpu", data_cfg=data_cfg, parallel_pool=None)
            feats_list = unbatch_feats(batch)
            for bi in range(len(batch["pdb_key"])):
                record_id = batch["pdb_key"][bi]
                record_id_to_feats[record_id] = feats_list[bi]

    # clean up processed structures
    shutil.rmtree(processed_struct_dir)

    return record_id_to_feats


if __name__ == "__main__":
    main()