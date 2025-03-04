import pickle
import shutil
from pathlib import Path
from typing import List

import biotite.structure.io as strucio
import numpy as np
import torch
import tqdm

from esm3.esm.sdk.api import ESMProtein


def create_esm3_embeddings(vqvae_encoder,
                           pdb_paths: List[str],
                           out_dir: str,
                           device: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    temp_dir = f"{out_dir}/temp"
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    embedding_dict = {}
    with torch.no_grad():
        for pdb in tqdm.tqdm(pdb_paths, desc="Running ESM3 on PDBs", leave=False):
            try:
                name = pdb[pdb.rfind("/") + 1:]
                if name[-4:] == ".pdb":
                    name = name[:-4]
                elif name[-4:] == ".cif":
                    name = name[:-4]
                    # convert cif to pdb in temp dir
                    structure = strucio.load_structure(pdb)
                    temp_pdb = f"{temp_dir}/{name}.pdb"
                    strucio.save_structure(temp_pdb, structure)
                    pdb = temp_pdb

                protein = ESMProtein.from_pdb(pdb)
                coords = protein.coordinates.unsqueeze(0).to(device)
                z_q, min_encoding_indices = vqvae_encoder.encode(coords = coords)

                embedding_dict[name] = (z_q.to("cpu"), min_encoding_indices.to("cpu"))

                del z_q, min_encoding_indices, coords
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print("error at ", pdb)
                print(e)
                continue

            # Dump each embedding to a pickle file
            out_file = Path(out_dir) / f"{name}.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(embedding_dict[name], f)

    # Delete temp dir
    shutil.rmtree(temp_dir)


def load_esm3_embeddings(embedding_paths: List[str]):
    all_embeds = []
    lengths = []
    for embedding_path in embedding_paths:
        with open(embedding_path, "rb") as f:
            embed = pickle.load(f)
        all_embeds.append(embed[0][0].mean(-2).numpy())
        lengths.append(embed[0][0].shape[-2])
    return np.array(all_embeds), lengths
