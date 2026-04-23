# ligand_validity — augment metadata with RCSB ligand-validity scores

A 4-stage pipeline that fills the `q_pn_unit_ligand_validity` column of
`metadata.parquet`. The augment step is parallelized across a 32-shard SLURM
array (~25-30x wallclock speedup over the legacy single-task path).

## Stages

| # | sbatch | Time | Role |
|---|---|---|---|
| 1 | `fetch.sbatch` | ~12 h, 8-task array | RCSB GraphQL → per-PDB JSON cache (network-bound). Only needed for a fresh dataset |
| 2 | `consolidate_cache.sbatch` | ~10 min, 1 task | per-PDB JSON cache → single `ligand_validity_cache.parquet` |
| 3 | `augment_array.sbatch` | ~30 min, 32-task array | augment `pdb_ids[id::32]`, write per-shard `(row_idx, value)` parquet |
| 4 | `finalize.sbatch` | ~10 min, 1 task | merge shard parquets → final `metadata_ligval.parquet` |

Data flow:
```
metadata.parquet ──┬──► fetch.sbatch ──► ligand_validity_cache_json/
                   │                              │
                   │                              ▼
                   │             consolidate_cache.sbatch
                   │                              │
                   │                              ▼
                   │                  ligand_validity_cache.parquet
                   │                              │
                   └──► augment_array.sbatch ◄────┘
                                  │
                                  ▼
                   ligand_validity_augment_shards/
                                  │
                                  ▼
                            finalize.sbatch
                                  │
                                  ▼
                       metadata_ligval.parquet
```

## How to run

All sbatch files hard-code `DATASET_DIR=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9`.
For a different dataset, edit the `DATASET_DIR=` line in each sbatch.

```bash
cd /home/users/zhkim216/code/allatom-design
SCRIPTS=scripts/sherlock_scripts/jinho/preprocessing/atomworks/ligand_validity
```

### Case A — `cache.parquet` already exists (most common)

```bash
JOB=$(sbatch --parsable ${SCRIPTS}/augment_array.sbatch)
sbatch --dependency=afterok:${JOB} ${SCRIPTS}/finalize.sbatch
```

### Case B — JSON cache exists but `cache.parquet` does not

Stage 2 is pure I/O, so an interactive node is usually faster than queueing:

```bash
sh_dev -t 30 -m 8GB
cd /home/users/zhkim216/code/allatom-design
DATASET=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9
python3 -m allatom_design.data.preprocessing.atomworks.ligand_validity.consolidate_cache \
    --cache-dir     ${DATASET}/ligand_validity_cache_json \
    --cache-parquet ${DATASET}/ligand_validity_cache.parquet
exit
```

Then proceed with Case A.

To stay fully in batch:
```bash
J1=$(sbatch --parsable ${SCRIPTS}/consolidate_cache.sbatch)
J2=$(sbatch --parsable --dependency=afterok:${J1} ${SCRIPTS}/augment_array.sbatch)
sbatch --dependency=afterok:${J2} ${SCRIPTS}/finalize.sbatch
```

### Case C — fresh dataset (start from fetch)

```bash
J0=$(sbatch --parsable ${SCRIPTS}/fetch.sbatch)
J1=$(sbatch --parsable --dependency=afterok:${J0} ${SCRIPTS}/consolidate_cache.sbatch)
J2=$(sbatch --parsable --dependency=afterok:${J1} ${SCRIPTS}/augment_array.sbatch)
sbatch --dependency=afterok:${J2} ${SCRIPTS}/finalize.sbatch
```

Fetch can die mid-run if RCSB rate-limits — `--skip-existing` (default) means
just resubmitting picks up where it left off.

## Monitoring

```bash
squeue -u zhkim216
tail -f /scratch/users/zhkim216/job_output/data_preprocessing/ligval_augment_*_0.out  # shard 0 progress
ls ${DATASET_DIR}/ligand_validity_augment_shards/ | wc -l                              # completed shards (target: 32)
```

## Troubleshooting

- **Some shards failed** — resubmit only the failed ids: `sbatch --array=<id>[,<id>...] ${SCRIPTS}/augment_array.sbatch`
- **`finalize.sbatch` aborts on missing shards** — the missing ids are listed on stderr; resubmit those, then re-run finalize
- **All stages use atomic writes** (`tmp + os.replace`) — a killed job leaves no stale outputs behind
