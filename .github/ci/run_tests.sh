set -e  # Exit on error

echo "Running from $PWD"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
apptainer_path=/net/software/containers/users/ncorley/atomworks/atomworks_dev.sif
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
