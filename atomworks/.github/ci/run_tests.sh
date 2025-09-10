# set -e  # Exit on error

echo "Running from $PWD"

export CCD_MIRROR_PATH=/projects/ml/frozen_pdb_copies/2024_12_11_ccd
export PDB_MIRROR_PATH=/projects/ml/frozen_pdb_copies/2024_12_01_pdb
export TEMPLATE_LOOKUP_PATH=/projects/ml/TrRosetta/PDB-2021AUG02/list_v02.csv
export TEMPLATE_BASE_DIR=/projects/ml/TrRosetta/PDB-2021AUG02/torch/hhr/
export MSA_CACHE_PATH=/projects/ml/RF2_allatom/cache/msa
export AF2FB_PATH=/squash/af2_distillation_facebook
export X3DNA=/projects/ml/prot_dna/x3dna-v2.4
export RESIDUE_CACHE_DIR=/net/tukwila/ncorley/datahub/egret/egret_embeddings_ccd
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
apptainer_path=/projects/ml/modelhub/apptainer/modelhub_2025-07-07.sif
tar -xf /net/lab/pub/atomworks/test_pack_latest.tar.gz -C tests/data > /dev/null 2>&1

# Get max processes from environment variable, default to 24 if not set
N_CPU=${N_CPU:-24}
echo "Running tests with max $N_CPU CPUs"

# Ensure we can collect all tests (i.e. imports succeed)
echo "Testing imports by trying to collect all tests"
apptainer exec --bind $PWD:/workspace --pwd /workspace --env PYTHONPATH=/workspace/src $apptainer_path pytest -m "not benchmark" --collect-only tests/

# Run the tests in coverage mode (with 24 CPUs)
apptainer exec --bind $PWD:/workspace --pwd /workspace --env PYTHONPATH=/workspace/src $apptainer_path pytest -m "not benchmark" --cov=atomworks --cov-report=xml -n=auto --maxprocesses=$N_CPU --dist=worksteal tests/

# Require at least 80% coverage
apptainer exec --bind $PWD:/workspace --pwd /workspace --env PYTHONPATH=/workspace/src $apptainer_path coverage report --fail-under=80

# Output the coverage in a format GitLab can parse
apptainer exec --bind $PWD:/workspace --pwd /workspace --env PYTHONPATH=/workspace/src $apptainer_path coverage report | tail -n 1 | awk '{print "TOTAL", $NF}'
