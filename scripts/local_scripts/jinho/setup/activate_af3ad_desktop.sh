#!/bin/bash

# =============================================================================
# AF3AD Desktop Environment Activation Script
# =============================================================================
# This script activates the af3ad_desktop virtual environment with all
# GTX1080 compatibility settings.
#
# Usage: source scripts/local_scripts/jinho/setup/activate_af3ad_desktop.sh
# =============================================================================

VENV_PATH="$HOME/venv/af3ad_desktop"

if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please run the installation script first:"
    echo "  bash scripts/local_scripts/jinho/setup/install_af3ad_desktop.sh"
    return 1
fi

echo "Activating AF3AD desktop environment..."
source "$VENV_PATH/bin/activate"

echo "Environment activated. You can now run:"
echo "  python allatom_design/train_seq_denoiser.py --config-path configs_local/seq_denoiser --config-name debug_seq_denoiser_local.yaml"
