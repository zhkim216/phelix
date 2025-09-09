# import time
# from logging import getLogger

# import pytest
# import torch
# from torch.utils.data import DataLoader, Dataset

# from atomworks.ml.loaders.worksteal import WorkStealDataLoader

# logger = getLogger("atomworks.ml")


# class TestDataset(Dataset):
#     """Test dataset with configurable delays."""

#     def __init__(self, size: int = 1000, slow_freq: int = 10, super_slow_freq: int = 100):
#         self.size = size
#         self.slow_freq = slow_freq
#         self.super_slow_freq = super_slow_freq

#     def __len__(self):
#         return self.size

#     def __getitem__(self, idx):
#         if idx % self.slow_freq == 0:
#             time.sleep(0.3)  # Slow items
#         elif idx % self.super_slow_freq == 0:
#             time.sleep(2.0)  # Super slow items
#         else:
#             time.sleep(0.00001)  # Fast items
#         return torch.tensor([idx]), idx


# class ErrorDataset(Dataset):
#     def __len__(self):
#         return 10

#     def __getitem__(self, idx):
#         if idx == 5:
#             raise ValueError("Test error")
#         return idx


# def test_single_worker():
#     """Test single-threaded operation."""
#     dataset = TestDataset(size=20)
#     loader = WorkStealDataLoader(dataset, batch_size=4, num_workers=0, shuffle=False)

#     batches = list(loader)
#     assert len(batches) == 5

#     # Check batch contents
#     all_indices = []
#     for data, indices in batches:
#         all_indices.extend(indices.tolist())
#     assert sorted(all_indices) == list(range(20))


# def test_multi_worker():
#     """Test multi-process operation."""
#     dataset = TestDataset(size=40)
#     # (Use a generous timeout to avoid flakiness on CI)
#     loader = WorkStealDataLoader(dataset, batch_size=4, num_workers=2, shuffle=False, timeout=20.0)

#     with loader.worker_monitoring():
#         batches = list(loader)

#     assert len(batches) == 10

#     loader._shutdown_workers()


# def test_error_handling():
#     """Test error propagation from workers."""
#     dataset = ErrorDataset()
#     loader = WorkStealDataLoader(dataset, batch_size=2, num_workers=2)

#     with pytest.raises(ValueError):
#         list(loader)

#     loader._shutdown_workers()


# def test_persistence_fork():
#     """Test persistent workers."""
#     dataset = TestDataset(size=20, slow_freq=10000, super_slow_freq=10000)
#     loader = WorkStealDataLoader(
#         dataset, batch_size=4, num_workers=2, persistent_workers=True, multiprocessing_context="fork"
#     )

#     # First epoch
#     batches1 = list(loader)
#     assert len(batches1) == 5

#     # Second epoch - workers should still be alive
#     batches2 = list(loader)
#     assert len(batches2) == 5

#     # Explicit cleanup
#     loader._shutdown_workers()


# def test_persistence_with_spawn():
#     """Test persistent workers with 'spawn' context."""
#     dataset = TestDataset(size=20)
#     loader = WorkStealDataLoader(
#         dataset,
#         batch_size=4,
#         num_workers=2,
#         persistent_workers=True,
#         multiprocessing_context="spawn",
#     )

#     # First epoch
#     batches1 = list(loader)
#     assert len(batches1) == 5
#     assert loader._workers, "Workers should be initialized"
#     assert all(w.is_alive() for w in loader._workers), "All workers should be alive after first epoch"

#     # Second epoch
#     batches2 = list(loader)
#     assert len(batches2) == 5
#     assert all(w.is_alive() for w in loader._workers), "All workers should still be alive after second epoch"

#     # Explicit cleanup
#     loader._shutdown_workers()
#     assert not loader._workers, "Worker list should be empty after shutdown"


# @pytest.mark.slow
# def test_performance_speedup():
#     """Assess and demonstrate speedup from multi-worker loading."""
#     logger.info("\n--- Testing Performance Speedup ---")

#     # Test configuration
#     dataset_size = 400
#     # Imbalanced dataset: every 5th item is slow to load
#     dataset = TestDataset(size=dataset_size, slow_freq=5, super_slow_freq=100)

#     # --- Single-process (baseline) ---
#     logger.info("\nRunning with 0 workers (main process)...")
#     loader_single = WorkStealDataLoader(dataset, batch_size=4, num_workers=0, shuffle=False)
#     start_time_single = time.time()
#     list(loader_single)  # Consume all batches
#     end_time_single = time.time()
#     duration_single = end_time_single - start_time_single
#     logger.info(f"Single-process time: {duration_single:.2f}s")

#     # --- Multi-worker (standard dataloader) ---
#     logger.info("\nRunning with 4 workers (standard dataloader)...")
#     loader_multi = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)
#     # ... fetch first batch to warm up
#     loader_multi_iter = iter(loader_multi)
#     next(loader_multi_iter)
#     start_multi_standard = time.time()
#     for _ in loader_multi_iter:
#         pass
#     end_multi_standard = time.time()
#     duration_multi_standard = end_multi_standard - start_multi_standard
#     logger.info(f"Multi-worker time (standard dataloader): {duration_multi_standard:.2f}s")

#     # --- Multi-worker ---
#     num_workers = 4
#     logger.info(f"\nRunning with {num_workers} workers...")
#     loader_multi = WorkStealDataLoader(
#         dataset,
#         batch_size=4,
#         num_workers=num_workers,
#         shuffle=False,
#         max_queue_size=100,
#         timeout=20.0,  # Generous timeout for slower machines
#         persistent_workers=True,  # Use persistence to not include shutdown in timing
#     )
#     # ... fetch first batch to warm up
#     loader_multi_iter = iter(loader_multi)
#     next(loader_multi_iter)
#     start_multi_WorkSteal = time.time()
#     with loader_multi.worker_monitoring():
#         for _ in loader_multi_iter:
#             pass
#     end_multi_WorkSteal = time.time()
#     duration_multi_WorkSteal = end_multi_WorkSteal - start_multi_WorkSteal
#     logger.info(f"Multi-worker time ({num_workers} workers): {duration_multi_WorkSteal:.2f}s")

#     # Second iteration
#     start_multi_WorkSteal2 = time.time()
#     loader_multi_iter = iter(loader_multi)
#     with loader_multi.worker_monitoring():
#         for _ in loader_multi_iter:
#             pass
#     end_multi_WorkSteal2 = time.time()
#     duration_multi_WorkSteal2 = end_multi_WorkSteal2 - start_multi_WorkSteal2
#     logger.info(f"2nd iteration multi-worker time ({num_workers} workers): {duration_multi_WorkSteal2:.2f}s")

#     loader_multi._shutdown_workers()  # Clean up persistent workers

#     # --- Assertions ---
#     speedup_factor_standard = duration_single / duration_multi_standard
#     speedup_factor_WorkSteal = duration_single / duration_multi_WorkSteal
#     speedup_factor_WorkSteal_vs_standard = speedup_factor_WorkSteal / speedup_factor_standard
#     logger.info(f"\nSpeedup factor (single vs. standard-multi): {speedup_factor_standard:.2f}x")
#     logger.info(
#         f"Speedup factor (single vs. WorkSteal-muli): {speedup_factor_WorkSteal:.2f}x"
#     )  # This is the important one.
#     logger.info(f"Speedup factor (WorkSteal vs. standard-multi): {speedup_factor_WorkSteal_vs_standard:.2f}x")

#     # We expect a significant speedup. This can be flaky on CI,
#     # but is a good check for local validation.
#     assert speedup_factor_WorkSteal > 1.5


# if __name__ == "__main__":
#     pytest.main([__file__, "--verbose", "-s", "--log-cli-level=DEBUG"])
