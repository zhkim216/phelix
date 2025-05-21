# allatom-design
Repository for backbone generation + FAMPNN sequence design. For specific examples on usage, look at `sherlock_scripts/rshuai` for some batch scripts.

# Installation
This environment requires python 3.10 to run. On sherlock, you can simply activate the environment with:
```bash
ENV_DIR=/home/groups/possu/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate```
Contact Richard Shuai for detailed instructions on reproducing this environment.

# Datasets
Currently, a few different datasets are supported:
- `af3_pdb`: the dataset of PDBs from AlphaFold3, includes both monomers and interfaces. FAMPNN was trained on this dataset.
- `af3_pdb_monomer`: all single chains extracted from `af3_pdb`, including any that are part of multimeric complexes.
- `augmented_af3_monomer_v1`: FAMPNN1 ESMFold-predictions on `af3_pdb_monomer` using an older version of FAMPNN, with train lengths [32, 256] and eval lengths [32, 512]
- `augmented_af3_monomer_v2`: FAMPNN4 ESMFold-predictions on `af3_pdb_monomer` using the latest version of FAMPNN, with train lengths [32, 256] and eval lengths [32, 256]

We also have code for cluster-based stratified sampling of these datasets. For training atom denoisers, see `allatom_design/data/datasets/ad_dataset.py` for how to load examples in from these datasets. For training sequence denoisers, see `allatom_design/data/datasets/sd_dataset.py`.


# Backbone generation
## Unconditional sampling
The `allatom_design/eval/sampling/bb_unconditional.py` script can be used to sample backbones from the denoiser. This script can also optionally evaluate ESMFold self-consistency of the generated backbones, either with ProteinMPNN (default) or FAMPNN. You can also optionally run a separate script, `allatom_design/eval/run_sc_eval.py`, to evaluate self-consistency on any set of input PDBs.

# Sequence design

## FAMPNN sequence design
The `allatom_design/eval/sampling/fampnn_multi.py` script can be used to sample sequences from FAMPNN.

It takes in a `checkpoint_path` and `pdb_dir` and will sample a given number of sequences per PDB in the `pdb_dir`. Optionally, a `pdb_name_list` can be provided to sample from a subset of PDBs in the `pdb_dir`.

It also supports positional constraints on the generated sequences by including a CSV with any number of the following columns (make sure the header is named correctly):
- `pdb_name`: name of the PDB to constrain sampling on
- `fixed_pos_seq`: comma-separated list of <chain + residue indices> to fix the sequence from the input PDB at given positions, e.g. "A1-100,B1-100"
- `fixed_pos_scn`: comma-separated list of <chain + residue indices> to fix the sidechain from the input PDB at given positions, e.g. "A1-100"
- `fixed_pos_override_seq`: comma-separated list of <chain + residue indices> to fix at given sequence, overriding the sequence in the PDB, e.g. "A26:A,A27:L"
- `pos_restrict_aatype`: comma-separated list of <chain + residue indices>:<allowed 1-letter aatypes>, e.g. "A26:AVG,A27:VG"

e.g.
```
pdb_name,fixed_pos_seq,fixed_pos_scn,fixed_pos_override_seq,pos_restrict_aatype
2fyzA,"A1-25,B1-25","A1-25","B26:A,B27:L","A30:AVG,A31:VG"
```

## Sidechain packing
The `allatom_design/eval/sampling/sidechain_pack.py` script can be used to sample sidechains from FAMPNN given a fixed sequence.

# Training
## Backbone model training
The entry point for training the backbone model is `allatom_design/train_atom_denoiser.py`. See `allatom_design/configs/atom_denoiser.yaml` for the default configuration.

### Model checkpoint evaluation
Once a model is trained, you can run `allatom_design/eval/eval_bb_gen_training.py` to evaluate self-consistency on sampled backbones from checkpoints along the training run.

For motif scaffolding models, there is also a `allatom_design/eval/eval_scaffold_training.py` script that can be used to evaluate self-consistency on PDBs assuming the same distribution of randomly sampled motifs as during training.

## Sequence design model training
The entry point for training the sequence design model is `allatom_design/train_seq_denoiser.py`. See `allatom_design/configs/seq_denoiser.yaml` for the default configuration.

