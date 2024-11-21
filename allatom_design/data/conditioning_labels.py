"""
Conditioning label information.
"""
from typing import Dict, Optional

import torch
from torchtyping import TensorType

# Map from token to ID for each conditioning type
TOKEN_TO_ID = {
    "crop_aug": {
        "UNCROPPED": 0,
        "CROPPED": 1,
    },
    "designability": {
        "UNDESIGNABLE": 0,
        "DESIGNABLE": 1
    }
}

# Number of classes for each conditioning type
COND_NUM_CLASSES = {k: len(v) for k, v in TOKEN_TO_ID.items()}

# Map from conditioning type to default token
DEFAULT_TOKEN = {
    "crop_aug": "UNCROPPED",
}

# Map from conditioning type to default token ID
DEFAULT_TOKEN_ID = {k: TOKEN_TO_ID[k][DEFAULT_TOKEN[k]] for k in TOKEN_TO_ID if k in DEFAULT_TOKEN}


# Helpers
def create_cond_labels_input(batch_size: int,
                             cond_labels: Optional[Dict[str, str]],
                             device: torch.device
                             ) -> Dict[str, TensorType["b", int]]:
    """
    Create conditioning labels for the batch.
    """
    if cond_labels is None:
        return {}
    return {k: torch.tensor([TOKEN_TO_ID[k][cond_labels[k]]] * batch_size, dtype=torch.int64, device=device) for k in cond_labels}
