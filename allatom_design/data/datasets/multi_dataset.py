from torch.utils.data import Dataset
import random
from typing import List
import torch

class MultiDataset(Dataset):
    def __init__(self, datasets: List[Dataset], dataset_weights: List[float], primary_dset_idx: int = 0):
        self.datasets = datasets
        self.dataset_probabilities = dataset_weights
        self.primary_dset_idx = primary_dset_idx
        self.primary_dataset_length = len(datasets[primary_dset_idx])

        assert sum(dataset_weights) == 1.0, f"Dataset weights sum to {sum(dataset_weights)}, but should sum to 1.0"

        self.primary_indices = list(range(self.primary_dataset_length))
        random.shuffle(self.primary_indices)

    def __len__(self):
        return self.primary_dataset_length

    def __getitem__(self, idx):
        # Decide from which dataset to sample based on probabilities
        dset_idx = torch.multinomial(torch.tensor(self.dataset_probabilities), num_samples=1).item()

        if dset_idx == self.primary_dset_idx:
            # Use the current index from the primary dataset
            sample_idx = self.primary_indices[idx]
        else:
            # Sample randomly from the other datasets (with replacement)
            sample_idx = random.randint(0, len(self.datasets[dset_idx]) - 1)

        return self.datasets[dset_idx][sample_idx]
