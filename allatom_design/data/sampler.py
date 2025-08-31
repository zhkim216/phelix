from typing import Iterator

import numpy as np


class Sampler:
    def __init__(self, weights: np.ndarray):
        self.weights = weights / weights.sum()

    def sample(self, random: np.random.RandomState) -> Iterator[int]:
        """
        Sample indices from the dataset infinitely.

        Args:
            random: Random state to use for sampling.

        Returns:
            Iterator[int]: Iterator of sampled indices.
        """
        while True:
            idx = random.choice(len(self.weights), p=self.weights)  # O(n) per draw
            yield idx
