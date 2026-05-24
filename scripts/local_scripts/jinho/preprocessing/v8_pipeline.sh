#!/bin/bash
# v8 metadata preprocessing pipeline (local execution).
#
# Pipeline order (v3 debug/260205 pattern):
#   step0: CCD filtering + SMILES cache (threshold-independent)
#   step1: cluster_sequences @ SEQ_ID (adds q_pn_unit_is_protein + q_pn_unit_cluster_id)
#   step2: BMSM augment (reads q_pn_unit_is_protein from step1; uses SMILES cache for heavy atoms)
#   step3: complete-linkage ligand cluster (adds q_pn_unit_bmsm_ligand_cluster_id)
#
# Input:  /home/possu/jinho/datasets/atomworks_pdb_full_v8/metadata_ligval.parquet
# Output: .../metadata_ligval_seq_clustered_${SEQ_ID_TAG}_bmsm_ligclust.parquet
#
# Usage:
#   bash scripts/local_scripts/jinho/preprocessing/v8_pipeline.sh              # full pipeline
#   bash scripts/local_scripts/jinho/preprocessing/v8_pipeline.sh step1        # run only step1
#   bash scripts/local_scripts/jinho/preprocessing/v8_pipeline.sh step2,step3  # step2 then step3
#   SEQ_ID=0.3 bash scripts/local_scripts/jinho/preprocessing/v8_pipeline.sh step1
#       # override default threshold (0.4 matches cluster_sequences.yaml; LMPNN ref is 0.3)
#
# Steps are idempotent only at the cluster_sequences level (mmseqs subdir reuse).
# Rerun a step by deleting its output parquet or passing it explicitly.

set -euo pipefail

ROOT=/home/possu/jinho/datasets/atomworks_pdb_full_v8
CODE=/home/possu/jinho/allatom-design
PLINDER_TXT=/home/possu/jinho/datasets/ccd_reference_lists/plinder_artifact_ccd_codes.txt

# Sequence-identity threshold for cluster_sequences. Env override:
#   SEQ_ID=0.3 bash v8_pipeline.sh step1
# Tag is auto-derived via bash parameter expansion (0.4 -> "04") so tag can
# never drift from the value.
SEQ_ID="${SEQ_ID:-0.4}"
SEQ_ID_TAG="${SEQ_ID//./}"

SUFFIX=v8

# Intermediate parquet naming follows the pipeline order:
#   metadata_ligval -> + seq_clustered_{TAG} -> + bmsm -> + ligclust
INPUT_PARQUET="$ROOT/metadata_ligval.parquet"
SEQ_CLUST_PARQUET="$ROOT/metadata_ligval_seq_clustered_${SEQ_ID_TAG}.parquet"
BMSM_PARQUET="$ROOT/metadata_ligval_seq_clustered_${SEQ_ID_TAG}_bmsm.parquet"
LIGCLUST_PARQUET="$ROOT/metadata_ligval_seq_clustered_${SEQ_ID_TAG}_bmsm_ligclust.parquet"

STEPS="${1:-step0,step1,step2,step3}"

run_step() {
    local step="$1"
    case "$step" in
        step0)
            echo "=== Step 0: passed_ccd_codes_metadata_${SUFFIX}.txt + ccd_smiles_cache_metadata_${SUFFIX}.json ==="
            python3 -m allatom_design.data.preprocessing.atomworks.bmsm.ccd_filter \
                --metadata-parquet "$INPUT_PARQUET" \
                --suffix "$SUFFIX" \
                --plinder-artifact-txt "$PLINDER_TXT" \
                --output-dir "$ROOT"
            ;;
        step1)
            echo "=== Step 1: cluster_sequences @ $SEQ_ID -> $SEQ_CLUST_PARQUET ==="
            python3 -m allatom_design.data.preprocessing.atomworks.cluster_sequences \
                pdb_path="$ROOT" \
                parquet_path="$INPUT_PARQUET" \
                seq_id_threshold="$SEQ_ID"
            ;;
        step2)
            echo "=== Step 2: BMSM augment -> $BMSM_PARQUET ==="
            python3 -m allatom_design.data.preprocessing.atomworks.bmsm.augment_metadata_with_bmsm \
                --input-parquet "$SEQ_CLUST_PARQUET" \
                --output-parquet "$BMSM_PARQUET" \
                --passed-ccd-codes-txt "$ROOT/passed_ccd_codes_metadata_${SUFFIX}.txt" \
                --smiles-cache-json "$ROOT/ccd_smiles_cache_metadata_${SUFFIX}.json" \
                --heavy-atom-cache-json "$ROOT/ccd_heavy_atom_counts_${SUFFIX}.json"
            ;;
        step3)
            echo "=== Step 3: complete-linkage ligand cluster -> $LIGCLUST_PARQUET ==="
            python3 -m allatom_design.data.preprocessing.atomworks.bmsm.cluster_ligands \
                --input-parquet "$BMSM_PARQUET" \
                --output-parquet "$LIGCLUST_PARQUET" \
                --smiles-cache-json "$ROOT/ccd_smiles_cache_metadata_${SUFFIX}.json" \
                --tanimoto-cutoff 0.8
            ;;
        *)
            echo "Unknown step: $step" >&2
            exit 2
            ;;
    esac
}

IFS=',' read -ra STEP_ARR <<< "$STEPS"
for s in "${STEP_ARR[@]}"; do
    run_step "$s"
done

echo "=== Done ($STEPS) ==="
