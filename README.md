# allatom-design

## Current progress for context

I've finished a basic pass through data processing / loading for a train loop. The data preprocessing scripts are in [here](https://github.com/ProteinDesignLab/allatom-design/tree/rshuai/atomworks/allatom_design/data/preprocessing/atomworks) (with corresponding configs in under `allatom_design/configs/data/preprocessing/atomworks`). The basic pipeline is:

1. `build_metadata_parquet.py` - Reads in CIFs and extracts per-chain info, including contacting chains for defining interfaces later
2. `cluster_sequences.py` - reads in a parquet file from (1) and runs mmseqs2 clustering, adding cluster IDs as a column to the parquet and saving to `metadata_clustered.parquet`
3. `preprocess_examples.py` - using a parquet from (1), loads in cifs, preprocesses them with `allatom_design/data/transform/preprocess.py` / `preprocess_transform()`, and saves examples out to a `cached_examples` dir.

In `atomworks_sd_dataset.py` [here] (https://github.com/ProteinDesignLab/allatom-design/blob/rshuai/atomworks/allatom_design/data/datasets/atomworks_sd_dataset.py):

1. We load in the chain parquet with pandas, build a df describing interfaces, and determine sampling weights based on cluster sizes and chain types.
2. Then, we load in a preprocessed example from the cache.
3. Run the `sd_featurizer` ([here](https://github.com/ProteinDesignLab/allatom-design/blob/rshuai/atomworks/allatom_design/data/transform/sd_featurizer.py)) transforms to handle cropping and featurization into features that look a lot like boltz feats.

I haven't implemented filtering by e.g. resolution yet, but should be able to somewhat mock the `PandasDataset` class from atomworks for this (we just have to handle the interface df properly as well).  Similar to my old code, the `atomworks_sd_dataset.py` configurations can be seen in `seq_denoiser.yaml` under data: [here](https://github.com/ProteinDesignLab/allatom-design/blob/rshuai/atomworks/allatom_design/configs/seq_denoiser/seq_denoiser.yaml)


## Environment setup

### Apptainer
See `scripts/sherlock_install.sh` for a script to install the `uv` environment into your `$SCRATCH` using the apptainer image. Then, for all future runs, just set the environment variables and launch using the apptainer, and you should be good to go.

Environment variables:
```bash
export IMG="$GROUP_HOME/containers/pytorch_25.08.sif"
export REPO_DIR="/home/users/rshuai/code/allatom-design"  # change this to your own repo path
export ENV_DIR="$SCRATCH/envs"
```

To test that the environment is set up correctly, you can do:
```bash
apptainer exec --nv --bind $HOME,$SCRATCH,$GROUP_HOME,$REPO_DIR,$ENV_DIR \
  $IMG \
  bash -lc 'source '"$ENV_DIR"'/allatom_design/bin/activate && cd '"$REPO_DIR"' && python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
```

For example, to run an interactive session within the apptainer, make sure the corresponding environment variables are set, and you can do:

```bash
apptainer exec --nv \
  --bind $HOME,$SCRATCH,$GROUP_HOME,$REPO_DIR,$ENV_DIR \
  $IMG \
  bash -lc 'source '"$ENV_DIR"'/allatom_design/bin/activate && cd '"$REPO_DIR"' && exec bash --noprofile --norc -i'
```


If you need to submit an sbatch job, please ask GPT for help for now. I will look into writing a wrapper so you can do something like `submit_sbatch_in_apptainer.sh <sbatch_script_name>` to submit a job, but don't have time to do this right now.

### Environment variables
`atomworks` also requires you to set certain environment variables before running any script that imports it. Here's an example of my `launch.json` for debugging a random PDB.

Notice `PDB_MIRROR_PATH` and `CCD_MIRROR_PATH`. For `CCD_MIRROR_PATH`, you'll need to use `scripts/get_ccd_mirror.sh` to get that. If you want to run `cluster_sequences.py`, you'll also need mmseqs installed in `SOFTWARE_PATH`, which is called like `{os.environ['SOFTWARE_PATH']}/mmseqs/bin/mmseqs`

```json
      {
        "name": "seq_des_single",
           "type": "python",
           "request": "launch",
           "program": "/home/rshuai/research/huang_lab/allatom-design/allatom_design/eval/sampling/seq_des_single.py",
           "args": [
            "ckpt_path=/home/rshuai/research/huang_lab/allatom-design/out_dir/train_seq_denoiser/allatom_design/debug/checkpoints/ema/sd-step10-epoch00-ema0.99.ckpt",
            "pdb_path=/home/rshuai/research/huang_lab/allatom-design/out_dir/hyejin_fix_chain_name/100_2_minimized.pdb",
            "run_self_consistency_eval=false",
            "sampling_cfg_overrides.batch_size=1",
            "sampling_cfg_overrides.num_seqs_per_pdb=3",
            "out_dir=out_dir/seq_des_single/debug_atomworks",
            "struct_pred_cfg.model_name=boltz1",
            "num_workers=1"
           ],
           "env": {
               "SOFTWARE_PATH": "/media/scratch/software",
               "PDB_MIRROR_PATH": "",
               "CCD_MIRROR_PATH": "/media/scratch/datasets/ccd",
             },
           "console": "integratedTerminal",
           "justMyCode": true
      },
```


### What I need help with (for Tianyu)
I think the main challenge remaining is certain aspects of on-the-fly loading for sampling, and handling cif saving. I started with moving over `seq_des_single.py` for sampling on a single pdb/cif, but I haven't touched the actual `run_seq_des()` function from `seq_des_utils.py`.

I need help loading raw cifs and preprocessing them appropriately through `get_sd_batch()`, but I don't know how to with atomworks. Their dataloading at train time discards `auth_seq_id` at some deep point within their code, and I think it'd require copying over large parts of their code to modify. Alternatively I remember a recommendation in their code to use their `load_any()` in `io_utils` with `extra_fields=["all"]`, but not sure if that's the easiest way yet since it doesn't come with the necessary transforms (in training, we need `preprocess_transform()` followed by `sd_featurizer()`).

And cif saving also seems a little tough. Their main function only takes in biotite-like atomarrays, so if you change the sequence / restype during seq design, you need to manually handle which atoms to save. Extra complicated when trying to condition on stuff, or eventually for handling residue indices for scaffolding. The simplifying assumption I've made for sequence design before is that the model only sees ground truth for tokens specified as 1 in `seq_cond_mask` and atoms specified as 1 in `atom_cond_mask` at sampling time, so that's what we use to save coordinates.

Maybe less important, but I also noticed that after train-time cropping, cif saving gets a little glitchy, where the SEQRES does not line up with the label seq id, so PyMOL renders it strangely by reading the first e.g. 21 unresolved residues from SEQRES, then reading the residue identities from the actual structure, causing a duplication of the first few residues.
