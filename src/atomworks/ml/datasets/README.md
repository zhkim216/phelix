# Datasets and Samplers Overview

This README provides an overview of how to handle, extend, and add new datasets for training structure-based models utilizing the `atomworks.ml` and `atomworks.io` repositories.
It's still a work-in-progress; additional contributions (to the codebase, AND the README) welcome - and encouraged!

## How do I add a new dataset for training?

There are a few steps to add a new dataset for training.

### Step 1: Create the dataframe on disk

This step will proceed differently depending on the nature of your desired dataset.

#### Case A: Starting from scratch

As long as you have a directory containing all of your structure files, you're good to go!

The dataframe construction process is described in greater detail at `scripts/preprocessing/README.md`, which should be the primary resource for this process. In brief, you will perform the following sub-steps:

1. Convert structure files to individual CSV files
2. Combine these CSV files into a pn_units dataframe
3. (Optional): Performing clustering on the pn_units DataFrame
4. Construct a corresponding interfaces DataFrame

For a small dataset that does not require parallel processing, a simple clustering-free workflow might look like this:

```python
from pathlib import Path
from scripts.preprocessing.pdb import get_csvs_from_structures
from scripts.preprocessing.pdb.generate_pn_units_df import generate_pn_units_df
from scripts.preprocessing.pdb.generate_interfaces_df import generate_and_save_interfaces_df

# Declare paths and structure file extension
INPUT_DIR = Path("/path/to/input/structures")
OUTPUT_DIR = Path("/path/to/output/dataframes")
STRUCTURE_FILE_EXTENSION=".pdb"

# Get csv files for each structure
get_csvs_from_structures.run_pipeline(base_dir=INPUT_DIR, out_dir = OUTPUT_DIR, from_rcsb = False, file_extension=STRUCTURE_FILE_EXTENSION)

# Combine the csv files into a single pn_units dataframe
generate_pn_units_df(OUTPUT_DIR / "csv", OUTPUT_DIR / "pn_units_df.parquet", num_workers=1, dataset_name = 'my_example_dataset')

# Generate the corresponding interfaces dataframe
generate_and_save_interfaces_df(OUTPUT_DIR / "pn_units_df.parquet", OUTPUT_DIR / "interfaces_df.parquet")
```

#### Case B: Subsetting the PDB

If your dataset is a subset of the PDB, this step is even faster! You simply need to subset the existing dataframe for the entire PDB, then adjust the `example_id` to be unique to your dataset.
By convention, the `example_id` values are generated with `atomworks.ml.common.generate_example_id`.

Dataframes for the full PDB are located on the `DIGS` at `/projects/ml/atomworks.ml/dfs/2024_12_01_pn_units_df.parquet` and `/projects/ml/atomworks.ml/dfs/2024_12_01_interfaces_df.parquet`.

As an example, to keep all `pn_units` from a specified list of PDB IDs:

```python
import pandas as pd
from atomworks.ml.common import generate_example_id

# MY_PDB_IDS = <list of pdb_ids to include>
pdb_pn_units = pd.read_parquet('/path/to/full_pn_units_df.parquet')

# Subset as desired, using any pandas operation
my_pn_units = pdb_pn_units[pdb_pn_units["pdb_id"].isin(MY_PDB_IDS)]

# Rework the example ids to prevent dataset ambiguity
my_pn_units["example_id"] = my_pn_units.apply(
    lambda x: generate_example_id(
        ["my_example_dataset", "pn_units"],
        x["pdb_id"],
        x["assembly_id"],
        [x["q_pn_unit_iid"]],
    ),
    axis=1,
)

# Save the output to disk
my_pn_units.to_parquet("/path/to/my_dataset.parquet")
```

**Important Note:** Cluster information will be carried over from the AF3-like clustering performed on the full PDB. If you would like to recluster your structures, you should first perform the desired subsetting on the full pn_units dataframe, then remove the cluster columns and continue from Case A, sub-step 3.

### Step 2 (optional): Add any templates or MSAs to the appropriate locations

If using the standard MSA loading transforms, for MSAs to be correctly recognized, we must:

- Move the MSA's to `/projects/msa` (if others may find them useful; not required)
- Rename the MSA files according to the SHA-256 hash of the query sequence (we MUST use the function provided at `atomworks.ml.utils.misc.hash_sequence`) (the scripts in `scripts/preprocessing/msa` may be helpful)
- Ensure that the msa directory is a **flat** directory (again, the scripts in `scripts/preprocessing/msa` may be helpful)
- Add the directory path to the `msa_dirs` argument for `LoadPolymerMSAs` within the `hydra` configuration

For templates, we don't currently support any way to add additional sources.

### Step 3: Update the main training YAML

Your new dataset is now ready to go! All that remains is telling the model to include it.

While different models will have their config files structured differently, any model using `atomworks.ml` to parse input structures will at some point instantiate `StructuralDatasetWrapper`(s). Let's suppose you've followed the steps above to create a dataset parquet file at `/path/to/my_dataset.parquet`. Assuming the model is using `hydra`, you should add something like this to the YAML file:

```yaml
my_dataset:
    _target_: atomworks.ml.datasets.datasets.StructuralDatasetWrapper
    dataset_parser:
        _target_: atomworks.ml.datasets.parsers.GenericDFParser
    dataset:
        _target_: atomworks.ml.datasets.datasets.PandasDataset
        name: my_dataset
        id_column: example_id
        data: /path/to/my_dataset.parquet
```

**Important Note:** Both the `StructuralDatasetWrapper` and the `PandasDataset` have additional arguments not listed in this example. Please see the relevant documentation to decide how you would like to set these other argumetns.

One other thing to consider is weighted dataset sampling. The default behavior is to sample each example with equal probability, but cluster and/or type info can often be used to construct better sampling schemes. While
different models will handle the config structure differently here, a few useful functions for computing a tensor of sampling weights can be found in `atomworks.ml.samplers`. For example, to compute weights as the inverse of the example's cluster size:

```yaml
weights:
    _target_: atomworks.ml.samplers.calculate_weights_by_inverse_cluster_size
    cluster_column_name: "cluster"
```

As always, if you develop a broadly-applicable sampling scheme that is not represented yet, please consider contributing to `atomworks.ml`!

And that's it -- happy training!

## What is the high-level sampling and data loading workflow?

At a high-level, dataloading occurs through the following steps:

1. Sample a dataset index to load via a Sampler, like standard PyTorch. Dataset indices represent examples in a dataframe that we want featurize and pass to the model for training. For example, the index "1" in the dataset "PDB Chains" might correspond to PDB ID "3ne2", biassembly "2", PN Unit (similar to a chain, see the `Glossary` in the `atomworks.io` `README`) "A_1".
2. Call `__getitem__` on the top-level dataset associated with the Torch `DataLoader` using the dataset index (e.g., "1"). There may be multiple datasets concatenated together, so we must identify which dataset the example came from to call the appropriate `__getitem__` method to load the dataframe row.
3. Given that dataset index (e.g., "1"), call `__getitem__` on the corresponding sub-dataset to retrieve the relevant row. For example, our row could be a `pd.Series` containing values: `{"pdb_id": "3ne2", "q_pn_unit_iid": "A_1", "assembly": "2", "extra_info": "proteins4ever"}`
4. (Within `load_example_from_metadata_row`) Parse the dataframe row into a standardized format we can proceed with. Although often trivial, we must standardize the outputs of individual dataset `__getitem__` before proceeding, build the path to the CIF/PDB file, and aggregate the appropriate information. For our example, we may use the `PNUnitsDFParser` (which inherits the `ABC` `MetadataRowParser`) to parse our `pd.Series` row into a dictionary like `{"example_id": "{pn_unit}{3ne2}{2}{A_1}", "path": "/somewhere/on/digs/3ne2.cif", "q_pn_units": "A_1"}`. Only the `example_id` and the `path` are required; however, some transformations must receive additional fiels (`q_pn_units` for cropping, for example).
5. (Within `load_example_from_metadata_row`) Given the path provided in the standardized dictionary, load the CIF/PDB file with `CIFUtils`.
6. (Within `load_example_from_metadata_row`) Combine the output of `CIFUtils` with the output from the `PNUnitsDFParser`.
7. Last, with our complete dictionary containing the output of `CIFUtils` and possibly additional fields from the `MetadataRowParser` (optional, depending on the `Transforms` used), execute the `Transforms` pipeline, and return the featurized results.

Whew, that was a lot of steps. Thankfully, most of that complexity is abstracted away and many users only need to concern themselves with small portions of the pipeline.

## How do we handle hierarchical dataset and sampling structures in Datahub?

We implemented a hierarchical dataset and sampling schema that can be extended indefinitely. Note that this setup is somewhat complex; most scenarios can be handled with a `StructuralDatasetWrapper` around a `PandasDataset`, or something similar.

### Example: Possible AF-3 Style Training Setup

For RF2AA, we have the following approximate dataset structure:

```plaintext
                 NamedConcatDataset
                        |
        ---------------------------------
        |                               |
 FB Distillation                NamedConcatDataset
(StructuralDatasetWrapper               |
 wrapping a PandasDataset)              |
                                        |
                                -----------------------
                                |                     |
                      Interfaces Dataset       PN Units Dataset
                    (StructuralDatasetWrapper  (StructuralDatasetWrapper
                     wrapping a PandasDataset)  wrapping a PandasDataset)
```

We may choose to implement a corresponding sampling schema like so:

```plaintext

                DistributedMixedSampler
                           |
                -------------------------
                |                       |
               0.2                     0.8
             Sampler1              MixedSampler
                                    /       \
                                   0.7       0.3
                                Sampler2   Sampler3

```

Then, with (0.8)(0.7) = (0.56) probability, we will draw an index corresponding to the Interfaces Dataset, with (0.2) probability we will draw from distillation, etc.

## Contributing

The initial version of this codebase was written by [Nate Corley](mailto:ncorley@uw.edu) and [Simon Mathis](mailto:simon.mathis@gmail.com) in the summer of 2024, in preparation for training an AF3-like structure prediction model. Additional preprocessing functionality was added by [Rafi Brent](mailto:rib7@uw.edu) in the fall of 2024.
