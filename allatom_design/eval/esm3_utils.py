import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
import tqdm
from esm3.esm.sdk.api import ESMProtein


def create_esm3_embeddings(vqvae_encoder,
                           pdb_paths: List[str],
                           out_dir: str,
                           device: str):
    embedding_dict = {}
    with torch.no_grad():
        for pdb in tqdm.tqdm(pdb_paths, desc="Running ESM3 on PDBs", leave=False):
            try:
                name = pdb[pdb.rfind("/") + 1:]
                if name[-4:] == ".pdb":
                    name = name[:-4]

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

    out_file = Path(out_dir) / "esm3_embed.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(embedding_dict, f)


def load_esm3_embeddings(pdbs: List[str], embed_path: str):
    all_embeds = []
    fp = Path(embed_path)
    with open(fp, "rb") as f:
        embed_dict = pickle.load(f)
    for k, v in embed_dict.items():
        if k in pdbs:
            all_embeds.append(v[0][0].mean(-2).numpy())
    return np.array(all_embeds)
