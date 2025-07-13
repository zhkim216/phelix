# allatom-design
Repository for backbone generation + FAMPNN sequence design. For specific examples on usage, look at `sherlock_scripts/rshuai` for some batch scripts.

# Installation
This environment requires python 3.10 to run. On sherlock, you can simply activate the environment with:
```bash
ENV_DIR=/home/groups/possu/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate
```

Contact Richard Shuai for detailed instructions on reproducing this environment.

# Datasets
Currently, we use a modified version of the Boltz-1 dataset for training:
- `boltz_v2`

For training atom denoisers, see `allatom_design/data/datasets/boltz_ad_dataset.py` for how to load examples in from these datasets. For training sequence denoisers, see `allatom_design/data/datasets/boltz_sd_dataset.py`.

# Training
## Backbone model training
The entry point for training the backbone model is `allatom_design/train_atom_denoiser.py`. See `allatom_design/configs/atom_denoiser.yaml` for the default configuration.

### Model checkpoint evaluation
Once a model is trained, you can run `allatom_design/eval/eval_bb_gen_training.py` to evaluate self-consistency on sampled backbones from checkpoints along the training run.

For motif scaffolding models, there is also a `allatom_design/eval/eval_scaffold_training.py` script that can be used to evaluate self-consistency on PDBs assuming the same distribution of randomly sampled motifs as during training.

## Sequence design model training
The entry point for training the sequence design model is `allatom_design/train_seq_denoiser.py`. See `allatom_design/configs/seq_denoiser.yaml` for the default configuration.

