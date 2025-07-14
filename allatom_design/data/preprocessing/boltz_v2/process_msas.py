#!/usr/bin/env python3
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig
from p_tqdm import p_umap
from redis import Redis
from tqdm import tqdm

from allatom_design.data.preprocessing.boltz_utils.a3m import parse_a3m
from allatom_design.eval.eval_utils.eval_setup_utils import start_redis


@hydra.main(config_path="../../../configs/data/preprocessing/boltz_v2", config_name="process_msas", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Process pre-downloaded MSAs from Boltz-1.
    """
    # Create dataset directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    redis_host, redis_port = "localhost", 7777
    start_redis(redis_host, redis_port, cfg.taxonomy_rdb_path)
    resource = MSAResource(host=redis_host, port=redis_port)

    # Fetch data
    print("Fetching data...")
    data = list(Path(cfg.raw_msa_dir).rglob("*.a3m*"))
    print(f"Found {len(data)} MSA's.")

    use_parallel = cfg.num_workers > 1

    # Run processing
    processed_msas_dir = f"{cfg.out_dir}/processed_msas"
    Path(processed_msas_dir).mkdir(parents=True, exist_ok=True)
    if use_parallel:
        # Create processing function
        fn = partial(
            process_msa,
            outdir=processed_msas_dir,
            max_seqs=cfg.max_seqs,
            resource=resource,
        )

        # Run in parallel
        p_umap(fn, data, num_cpus=cfg.num_workers)
    else:
        for path in tqdm(data):
            process_msa(
                path,
                outdir=processed_msas_dir,
                max_seqs=cfg.max_seqs,
                resource=resource,
            )


class MSAResource:
    """A shared resource for processing MSAs."""

    def __init__(self, host: str, port: int) -> None:
        """Initialize the redis database."""
        self._redis = Redis(host=host, port=port)

    def get(self, key: str) -> Any:  # noqa: ANN401
        """Get an item from the Redis database."""
        return self._redis.get(key)

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get an item from the resource."""
        out = self.get(key)
        if out is None:
            raise KeyError(key)
        return out


def process_msa(
    path: Path,
    outdir: str,
    max_seqs: int,
    resource: MSAResource,
) -> None:
    """Run processing in a worker thread."""
    outdir = Path(outdir)
    out_path = outdir / f"{path.stem}.npz"
    if not out_path.exists():
        msa = parse_a3m(path, resource, max_seqs)
        np.savez_compressed(out_path, **asdict(msa))


if __name__ == "__main__":
    main()
