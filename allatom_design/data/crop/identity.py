from dataclasses import replace
from typing import Optional

import numpy as np
from boltz.data.crop.cropper import Cropper
from boltz.data.types import Tokenized


class IdentityCropper(Cropper):
    """
    Identity cropper.
    Returns the original data unchanged.
    """

    def __init__(self) -> None:
        """Initialize the cropper."""
        pass

    def crop(  # noqa: PLR0915
        self,
        data: Tokenized,
        **kwargs
    ) -> Tokenized:
        """Identity cropper.

        Returns
        -------
        Tokenized
            The original data.
        """
        return data
