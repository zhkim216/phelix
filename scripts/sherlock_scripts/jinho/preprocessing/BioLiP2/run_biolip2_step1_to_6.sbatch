#!/bin/bash
#SBATCH --job-name=biolip2
#SBATCH -p possu
#SBATCH --output=/scratch/users/zhkim216/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/zhkim216/slurm_log_files/job_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# BioLiP2 Database Setup: Steps 1-6
# This script runs all prerequisite steps for download_pubmed.pl

set -e

BIOLIP_DIR="/scratch/users/zhkim216/mmCIF2BioLiP"
cd "$BIOLIP_DIR"

echo "=============================================="
echo "BioLiP2 Database Setup: Steps 1-6"
echo "=============================================="
echo "Start time: $(date)"
echo "Working directory: $BIOLIP_DIR"
echo ""

# Create directories
mkdir -p logs pubmed

# ============================================
# Step 1: Download PDB entries
# ============================================
echo "=== Step 1: Download PDB entries ==="
echo "Step 1 start: $(date)"

# Using rsyncPDB.sh for initial setup (faster)
./script/rsyncPDB.sh

echo "Step 1 completed: $(date)"
echo ""

# ============================================
# Step 2: Download SIFTS function annotation
# ============================================
echo "=== Step 2: Download SIFTS ==="
echo "Step 2 start: $(date)"

perl ./script/download_sifts.pl

# Optional: download CSA
perl ./script/download_csa.pl || echo "Warning: download_csa.pl failed, continuing..."

echo "Step 2 completed: $(date)"
echo ""

# ============================================
# Step 3: Download binding affinity
# (Requires Step 2)
# ============================================
echo "=== Step 3: Download binding affinity ==="
echo "Step 3 start: $(date)"

perl ./script/download_bind.pl || echo "Warning: download_bind.pl failed, continuing..."

perl ./script/download_moad.pl || echo "Warning: download_moad.pl failed, continuing..."

echo "Step 3 completed: $(date)"
echo ""

# ============================================
# Step 4: Download auxiliary data
# ============================================
echo "=== Step 4: Download auxiliary data ==="
echo "Step 4 start: $(date)"

perl ./script/download_ligand.pl
perl ./script/curate_smiles.pl

echo "Step 4 completed: $(date)"
echo ""

# ============================================
# Step 5: Convert mmCIF to PDB format
# (Requires Step 1)
# ============================================
echo "=== Step 5: Convert mmCIF to PDB ==="
echo "Step 5 start: $(date)"

perl ./script/curate_pdb.pl

echo "Step 5 completed: $(date)"
echo ""

# ============================================
# Step 6: Download PubMed abstracts
# (Requires Steps 2 and 5)
# ============================================
echo "=== Step 6: Download PubMed abstracts ==="
echo "Step 6 start: $(date)"

perl ./script/download_pubmed.pl

echo "Step 6 completed: $(date)"
echo ""

# ============================================
# Summary
# ============================================
echo "=============================================="
echo "BioLiP2 Steps 1-6 completed!"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Output directories:"
echo "  - PDB files: $BIOLIP_DIR/pdb/"
echo "  - Interim files: $BIOLIP_DIR/interim/"
echo "  - PubMed abstracts: $BIOLIP_DIR/pubmed/"
echo "  - Data files: $BIOLIP_DIR/data/"

