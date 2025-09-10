import pytest
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from atomworks.ml.datasets.datasets import ConcatDatasetWithID, FallbackDatasetWrapper, PandasDataset
from atomworks.ml.samplers import DistributedMixedSampler, FallbackSamplerWrapper, MixedSampler
from tests.ml.datasets.test_datasets import create_dummy_dataset


class FaultyDataset(PandasDataset):
    def __getitem__(self, idx):
        if idx in (5, 15):
            raise ValueError("Simulated error")
        return super().__getitem__(idx)


def test_fallback_dataset_wrapper():
    primary_dataset = create_dummy_dataset(length=10, name="primary")
    fallback_dataset = create_dummy_dataset(length=10, name="fallback")

    wrapped_dataset = FallbackDatasetWrapper(primary_dataset, fallback_dataset)

    # Test normal access (no fallback)
    for idx in range(len(primary_dataset)):
        assert all(wrapped_dataset[(idx, idx)] == primary_dataset[idx])

    # Test fallback access by raising an exception in the primary dataset
    faulty_primary_dataset = create_dummy_dataset(20, "faulty_primary", FaultyDataset)
    wrapped_dataset = FallbackDatasetWrapper(faulty_primary_dataset, fallback_dataset)

    for idx in range(len(faulty_primary_dataset)):
        if idx == 5:
            assert all(wrapped_dataset[(idx, idx)] == fallback_dataset[5]), 5
        elif idx == 15:
            assert all(wrapped_dataset[(idx, 2)] == fallback_dataset[2]), 15
        else:
            assert all(wrapped_dataset[(idx, idx)] == faulty_primary_dataset[idx]), idx


def test_fallback_sampler_wrapper():
    primary_dataset = create_dummy_dataset(length=20, name="primary", dataset_class=FaultyDataset)
    fallback_dataset = create_dummy_dataset(length=10, name="fallback")

    primary_sampler = SequentialSampler(primary_dataset)
    fallback_sampler = SequentialSampler(fallback_dataset)

    wrapped_sampler = FallbackSamplerWrapper(primary_sampler, fallback_sampler, n_fallback_retries=1)

    for primary_idx, fallback_idx in wrapped_sampler:
        assert primary_idx % 10 == fallback_idx, (primary_idx, fallback_idx)

    wrapped_dataset = FallbackDatasetWrapper(primary_dataset, fallback_dataset)

    for idxs in wrapped_sampler:
        if idxs[0] == 5:
            assert all(wrapped_dataset[idxs] == fallback_dataset[idxs[0]]), idxs
        elif idxs[0] == 15:
            assert all(wrapped_dataset[idxs] == fallback_dataset[idxs[1]]), idxs
        else:
            assert all(wrapped_dataset[idxs] == primary_dataset[idxs[0]]), idxs

    fallback_dataloader = DataLoader(wrapped_dataset, sampler=wrapped_sampler, collate_fn=lambda x: x, batch_size=1)

    for idx, example in enumerate(iter(fallback_dataloader)):
        if idx == 5 or idx == 15:
            assert all(example[0] == fallback_dataset[5])
        else:
            assert all(example[0] == primary_dataset[idx])


def test_distributed_mixed_sampler():
    dataset1 = create_dummy_dataset(length=25, name="1", dataset_class=FaultyDataset)
    dataset2 = create_dummy_dataset(length=25, name="2", dataset_class=FaultyDataset)
    dataset3 = create_dummy_dataset(length=25, name="3", dataset_class=FaultyDataset)

    # Samplers
    sampler_1 = SequentialSampler(dataset1)
    sampler_2 = SequentialSampler(dataset2)

    datasets_info_1_2 = [
        {"sampler": sampler_1, "dataset": dataset1, "probability": 0.9},
        {"sampler": sampler_2, "dataset": dataset2, "probability": 0.1},
    ]
    # First mixed sampler
    datasets_1_2_concat = ConcatDatasetWithID([dataset1, dataset2])
    mixed_sampler = MixedSampler(datasets_info=datasets_info_1_2, n_examples_per_epoch=None)
    # Second mixed sampler
    sampler_3 = SequentialSampler(dataset3)
    datasets_info_2 = [
        {"sampler": mixed_sampler, "dataset": datasets_1_2_concat, "probability": 0.5},
        {"sampler": sampler_3, "dataset": dataset3, "probability": 0.5},
    ]
    dataset_1_2_3_concat = ConcatDatasetWithID([datasets_1_2_concat, dataset3])
    dist_mixed_sampler_rank_0 = DistributedMixedSampler(
        datasets_info=datasets_info_2,
        n_examples_per_epoch=50,
        num_replicas=2,
        rank=0,
        shuffle=False,
    )

    dist_mixed_sampler_rank_1 = DistributedMixedSampler(
        datasets_info=datasets_info_2,
        n_examples_per_epoch=50,
        num_replicas=2,
        rank=1,
        shuffle=False,
    )

    # ==== Set up fallback logic ===
    fallback_dataset = create_dummy_dataset(length=1000, name="fallback")
    fallback_sampler_rank0 = RandomSampler(fallback_dataset)
    fallback_sampler_rank1 = RandomSampler(fallback_dataset)

    dataset = FallbackDatasetWrapper(dataset_1_2_3_concat, fallback_dataset)
    sampler_rank_0 = FallbackSamplerWrapper(dist_mixed_sampler_rank_0, fallback_sampler_rank0)
    sampler_rank_1 = FallbackSamplerWrapper(dist_mixed_sampler_rank_1, fallback_sampler_rank1)
    # ==============================

    dataloader_rank_0 = DataLoader(dataset, sampler=sampler_rank_0, collate_fn=lambda x: x, batch_size=1)
    dataloader_rank_1 = DataLoader(dataset, sampler=sampler_rank_1, collate_fn=lambda x: x, batch_size=1)

    iter_sampler_rank_0 = iter(sampler_rank_0)
    for example in dataloader_rank_0:
        idx = next(iter_sampler_rank_0)[0]
        if example[0]["example_id"] in fallback_dataset:
            assert idx in (5, 15, 25 + 5, 25 + 15, 25 * 2 + 5, 25 * 2 + 15)
        else:
            assert example[0]["example_id"] in dataset_1_2_3_concat

    iter_sampler_rank_1 = iter(sampler_rank_1)
    for example in dataloader_rank_1:
        idx = next(iter_sampler_rank_1)[0]
        if example[0]["example_id"] in fallback_dataset:
            assert idx in (5, 15, 25 + 5, 25 + 15, 25 * 2 + 5, 25 * 2 + 15)
        else:
            assert example[0]["example_id"] in dataset_1_2_3_concat


def test_multifallback_dataset():
    primary_dataset = create_dummy_dataset(length=10, name="primary")
    fallback_dataset = create_dummy_dataset(length=10, name="fallback")

    wrapped_dataset = FallbackDatasetWrapper(primary_dataset, fallback_dataset)

    # Test normal access (no fallback)
    for idx in range(len(primary_dataset)):
        assert all(wrapped_dataset[(idx, idx)] == primary_dataset[idx])

    # Test fallback access by raising an exception in the primary dataset
    faulty_primary_dataset = create_dummy_dataset(20, "faulty_primary", FaultyDataset)
    faulty_fallback_dataset = create_dummy_dataset(20, "faulty_fallback", FaultyDataset)
    wrapped_dataset = FallbackDatasetWrapper(faulty_primary_dataset, faulty_fallback_dataset)

    for idx in range(len(faulty_primary_dataset)):
        if idx == 5:
            assert all(wrapped_dataset[(idx, idx, idx + 1)] == faulty_fallback_dataset[6]), 5
        elif idx == 15:
            assert all(wrapped_dataset[(idx, 5, 15, 2)] == faulty_fallback_dataset[2]), 15
        else:
            assert all(wrapped_dataset[(idx, idx)] == faulty_primary_dataset[idx]), idx


if __name__ == "__main__":
    pytest.main([__file__])
