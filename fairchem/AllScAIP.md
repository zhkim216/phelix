# AllScAIP

Paper Link: [A recipe for scalable attention-based ML Potentials: unlocking long-range accuracy with all-to-all node attention](https://openreview.net/forum?id=1KSP0Ppiqw)

## Installation

First, clone the FAIR Chem repo with allscaip branch:

```bash
git clone -b allscaip https://github.com/EricZQu/fairchem.git
cd fairchem
```

Then, create a conda environment and install the dependencies:

```bash
conda create -n allscaip python=3.12
conda activate allscaip
pip install -e packages/fairchem-core[dev]
```

## Inference

Checkpoints are available at: [Google Drive Link](https://drive.google.com/drive/folders/1IbokwZtjV6e1bbJjaxunrqh3HdhvWdtl?usp=sharing). Three models are provided:
- `omol_all_sm_NeAnNoSi_ft_40E20F_fixed.pt`: Small model (35M) direct force pretrained on OMol 102M for 10 Epochs, and the conservative finetuned for 2 Epochs.
- `omol_all_d_md_NeNo_fixed.pt`: Medium model (85M) direct force pretrained on OMol 102M for 10 Epochs. (Not energy conserve!)
- `omol_all_d_md_cons_NeNo_fixed.pt`: Medium model (85M) fully conservative trained on OMol 102M for 10 Epochs.

You can use the `FAIRChemCalculator` to load a pretrained AllScAIP model and perform inference. Here's an example:

```python
from ase import units
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator

calc = FAIRChemCalculator.from_model_checkpoint("/path/to/your/checkpoint.pt", task_name="omol")

atoms = molecule("H2O")
atoms.calc = calc

dyn = Langevin(
    atoms,
    timestep=0.1 * units.fs,
    temperature_K=400,
    friction=0.001 / units.fs,
)
trajectory = Trajectory("my_md.traj", "w", atoms)
dyn.attach(trajectory.write, interval=1)
dyn.run(steps=1000)
```

## Training

To start training, you can use the provided configuration file `configs/allscaip/training`. Make sure to adjust the paths and parameters according to your setup.

For example, to train AllScAIP on the OMol 4M dataset, you first need to download the OMol dataset from Hugging Face: [OMol Dataset Link](https://huggingface.co/facebook/OMol25). For 4M, you need to download the [4M Training](https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train_4M.tar.gz) and [Validation](https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val.tar.gz) tarballs and extract them. 

Then, you need to modify the `omol_4M_train_path` and `omol_all_path` in the `configs/allscaip/training/cluster/local.yaml` file, such that `configs/allscaip/training/dataset/omol_4M.yaml` points to the correct training and validation paths. You should also modify the `run_dir` in `configs/allscaip/training/cluster/local.yaml` to your desired output directory.

Next, you need to modify the `job` section in `/home/ericqu/fairchem/configs/allscaip/training/omol_4m.yml`. For local testing, you need to set the `num_nodes` to 1. Also, change the `logger` field to your wandb account if you want to use wandb for logging.


Finally, you can start training with the following command:

```bash
farichem -c configs/allscaip/training/omol_4m.yml
```

(There are support for SLURM cluster training as well. Please change the `cluster` config file accordingly.)
