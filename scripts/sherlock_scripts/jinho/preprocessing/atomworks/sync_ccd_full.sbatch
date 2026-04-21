#!/bin/bash
#SBATCH --job-name=sync_ccd_full
#SBATCH -p possu,normal,owners
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/sync_ccd_full_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/sync_ccd_full_%j.err

source ~/.bashrc

mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing

echo "==================================="
echo "CCD Mirror Full Sync started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "==================================="

DEST="/scratch/users/zhkim216/datasets/ccd_mirror_v2"
REMOTE="rsync://ftp.ebi.ac.uk/pub/databases/msd/pdbechem_v2/ccd/"

mkdir -p $DEST

echo "Starting full CCD sync (~2GB from EBI PDBeChem)..."
rsync -rltvz --stats --no-perms --chmod=ug=rwX,o=rX \
    --delete --omit-dir-times \
    --include='*/' --include='*.cif' --exclude='*' \
    $REMOTE $DEST/

# Invalidate atomworks' .ccd_codes_cache so the next process does a fresh scan.
# rsync with --omit-dir-times does not bump the root dir mtime, so atomworks'
# `cache_mtime > dir_mtime` staleness check in
# atomworks/src/atomworks/io/utils/ccd.py:get_available_ccd_codes_in_mirror
# would otherwise keep serving the pre-sync code set.
rm -f "$DEST/.ccd_codes_cache"

echo ""
echo "==================================="
echo "Verification:"
echo "Total .cif files:"
find $DEST -name "*.cif" | wc -l
echo "Total size:"
du -sh $DEST
echo ""
echo "Sync completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "==================================="
echo ""
echo "Next steps:"
echo "  1. rm -rf /scratch/users/zhkim216/datasets/ccd_mirror"
echo "  2. mv /scratch/users/zhkim216/datasets/ccd_mirror_v2 /scratch/users/zhkim216/datasets/ccd_mirror"
