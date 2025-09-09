import atexit
import contextlib
import gc
import io
import os
import pickle
import queue
import signal
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import Any

import psutil
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Event, Queue
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data._utils.worker import WorkerInfo

logger = getLogger("atomworks.ml")

SAFETY_TIMEOUT = 10.0 * 60  # 10 minutes


class WorkerState(Enum):
    """Worker process states for monitoring."""

    INITIALIZING = "initializing"
    IDLE = "idle"
    LOADING = "loading"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class WorkerStats:
    """Lightweight statistics for monitoring worker performance."""

    items_processed: int = 0
    total_load_time: float = 0.0
    errors: int = 0
    state: WorkerState = WorkerState.INITIALIZING
    last_update: float = 0.0


class WorkStealDataLoader(DataLoader):
    """
    Drop-in replacement for torch.utils.data.DataLoader with work-stealing scheduling.

    Compatible with PyTorch's DataLoader API while providing better load
    balancing for datasets with highly variable loading times (e.g. where
    some rare items might take 10s of seconds to load while others finish in
    ms).

    Key Features:
        - Work-stealing scheduling for optimal load balancing
        - Automatic worker health monitoring and recovery
        - Configurable memory management with queue size limits
        - Support for persistent workers across epochs
        - Comprehensive error handling and logging
        - Performance statistics and monitoring

    Args:
        dataset: Dataset to load from
        batch_size: Number of samples per batch (default: 1)
        shuffle: Whether to shuffle data (default: False)
        sampler: Custom sampler (mutually exclusive with shuffle)
        batch_sampler: Custom batch sampler (mutually exclusive with batch_size,
            shuffle, sampler, and drop_last)
        num_workers: Number of worker processes (default: 0, use 0 for single-threaded)
        collate_fn: Function to collate samples into batches (default: default_collate)
        pin_memory: Whether to pin memory for CUDA transfers (default: False)
        drop_last: Whether to drop last incomplete batch (default: False)
        timeout: Timeout for getting batches in seconds (0 = no timeout, default: 0)
        worker_init_fn: Function to initialize workers (default: None)
        multiprocessing_context: Multiprocessing context - 'spawn', 'fork', or
            'forkserver' (default: 'spawn'). `Spawn` is recommended for most use
            cases as it places fewer restrictions on the dataset implementation.
        generator: Random generator for reproducibility (default: None)
        prefetch_factor: Number of batches to prefetch per worker (default: 2)
        persistent_workers: Whether to keep workers alive between epochs (default: False)
        max_queue_size: Maximum size of result queue in batches (default: None,
            uses prefetch_factor * num_workers)

    Raises:
        ValueError: If invalid arguments are provided (e.g., negative num_workers,
            conflicting sampler/shuffle settings, or persistent_workers with
            num_workers=0)

    Example:
        >>> loader = WorkStealingDataLoader(
        ...     dataset=my_dataset,
        ...     batch_size=32,
        ...     num_workers=4,
        ...     shuffle=True,
        ...     persistent_workers=True,
        ... )
        >>> for batch in loader:
        ...     # Process batch
        ...     pass
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None = 1,
        shuffle: bool = False,
        sampler: Sampler | None = None,
        batch_sampler: BatchSampler | None = None,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable | None = None,
        multiprocessing_context: str = "spawn",
        generator: torch.Generator | None = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        max_queue_size: int | None = None,
    ) -> None:
        # Validate arguments
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}")
        elif num_workers == 1:
            logger.warning(
                "WorkStealingDataLoader is not designed for single-worker loading. "
                "You should use a normal DataLoader instead, or use num_workers=0 or >1."
            )

        if batch_sampler is not None and (batch_size != 1 or shuffle or sampler is not None or drop_last):
            raise ValueError("batch_sampler is mutually exclusive with " "batch_size, shuffle, sampler, and drop_last")

        if sampler is not None and shuffle:
            raise ValueError("sampler is mutually exclusive with shuffle")

        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")

        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else 2
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.mp_context = multiprocessing_context
        # Warn if the user is using a non-spawn context
        if multiprocessing_context != "spawn":
            logger.warning(
                f"Using a non-spawn multiprocessing context ({multiprocessing_context})"
                " is not recommended for WorkStealingDataLoader. Using spawn instead."
            )
        self.generator = generator
        self.persistent_workers = persistent_workers

        # Set up collate function
        if collate_fn is None:
            if hasattr(dataset, "collate_fn"):
                collate_fn = dataset.collate_fn
            else:
                collate_fn = torch.utils.data._utils.collate.default_collate
        self.collate_fn = collate_fn

        # Replicate DataLoader's sampler logic to ensure .sampler and .batch_sampler attributes exist
        if batch_sampler is not None:
            self.sampler = batch_sampler.sampler
            self.batch_sampler = batch_sampler
        else:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
                else:
                    sampler = torch.utils.data.SequentialSampler(dataset)
            self.sampler = sampler
            self.batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last)

        # The work queue can be much larger than the result queue as it only
        # holds indices, so we set it to the smaller of 1M or the number of batches.
        self.max_work_queue_size = min(1_000_000, len(self.batch_sampler))

        # Calculate max *result* queue size
        if max_queue_size is None:
            # Heuristic: limit based on available memory
            available_memory = psutil.virtual_memory().available
            # Assume each batch uses at most 100MB (conservative)
            estimated_batch_memory = 100 * 1024 * 1024
            max_from_memory = int(available_memory * 0.5 / estimated_batch_memory)
            result_queue_size = min(
                max_from_memory,
                num_workers * prefetch_factor * 2 if num_workers > 0 else 10,
            )
        else:
            result_queue_size = max_queue_size
        self.max_result_queue_size = max(1, self.num_workers, result_queue_size)

        # Persistent worker management
        self._workers = []
        self._worker_stats = {}
        self._shutdown_event = None
        self._work_queue = None
        self._result_queue = None
        self._manager = None
        self._manager_process = None

        # Register cleanup
        self._weakref = weakref.ref(self)
        atexit.register(self._cleanup_workers_atexit, self._weakref)

        logger.info(
            f"Initialized WorkStealingDataLoader with {num_workers} workers, "
            f"prefetch_factor={prefetch_factor}, max_result_queue_size={self.max_result_queue_size}"
        )

    def _ensure_workers_initialized(self) -> None:
        """Create workers, queues, and manager if they don't exist."""
        if not self._workers:
            ctx = mp.get_context(self.mp_context)
            self._manager = ctx.Manager()
            self._work_queue = ctx.Queue(maxsize=self.max_work_queue_size)
            self._result_queue = ctx.Queue(maxsize=self.max_result_queue_size)
            self._shutdown_event = ctx.Event()
            self._worker_stats = self._manager.dict()
            self._init_workers(ctx)

    @staticmethod
    def _cleanup_workers_atexit(dataloader_ref: weakref.ReferenceType["WorkStealDataLoader"]) -> None:
        """Cleanup function called at exit."""
        dataloader = dataloader_ref()
        if dataloader is not None:
            dataloader._shutdown_workers()

    def _init_workers(self, ctx: mp.Context) -> None:
        """Initialize worker processes and queues."""
        # Get the manager's process for liveness checks
        # This is an internal detail, but necessary for robust shutdown
        if hasattr(self._manager, "_process"):
            self._manager_process = self._manager._process

        if self.mp_context == "spawn":
            # ... pre-pickle the dataset, worker_init_fn, and collate_fn to avoid re-pickling for each worker
            dataset_buffer, worker_init_fn_buffer, collate_fn_buffer = io.BytesIO(), io.BytesIO(), io.BytesIO()
            pickle.dump(self.dataset, dataset_buffer)
            pickle.dump(self.worker_init_fn, worker_init_fn_buffer)
            pickle.dump(self.collate_fn, collate_fn_buffer)

        # Start workers
        for worker_id in range(self.num_workers):
            logger.info(f"Starting worker {worker_id+1} of {self.num_workers} ({worker_id=})")
            self._worker_stats[worker_id] = WorkerStats()

            if self.mp_context == "spawn":
                # ... get the pickled values
                dataset_buffer.seek(0)
                worker_init_fn_buffer.seek(0)
                collate_fn_buffer.seek(0)
                dataset_maybe_pickled = dataset_buffer.getvalue()
                worker_init_fn_maybe_pickled = worker_init_fn_buffer.getvalue()
                collate_fn_maybe_pickled = collate_fn_buffer.getvalue()
            else:
                # ... use the values directly (fork context)
                dataset_maybe_pickled = self.dataset
                worker_init_fn_maybe_pickled = self.worker_init_fn
                collate_fn_maybe_pickled = self.collate_fn

            worker = ctx.Process(
                target=self._worker_loop,
                args=(
                    worker_id,
                    dataset_maybe_pickled,
                    self._work_queue,
                    self._result_queue,
                    self._shutdown_event,
                    collate_fn_maybe_pickled,
                    worker_init_fn_maybe_pickled,
                    self._worker_stats,
                    self.timeout,
                ),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        if self.mp_context == "spawn":
            # ... cleanup any buffers / pre-pickled values
            dataset_buffer.close()
            worker_init_fn_buffer.close()
            collate_fn_buffer.close()
            del dataset_buffer, worker_init_fn_buffer, collate_fn_buffer
            del dataset_maybe_pickled, worker_init_fn_maybe_pickled, collate_fn_maybe_pickled
            gc.collect()

        logger.info(f"Started {self.num_workers} worker processes")

    def _shutdown_workers(self) -> None:
        """Shutdown all worker processes gracefully."""
        if not self._workers:
            return

        logger.info("Shutting down workers...")

        # Signal shutdown
        if self._shutdown_event:
            self._shutdown_event.set()

        # Send sentinel values
        if self._work_queue:
            for _ in range(len(self._workers)):
                with contextlib.suppress(Exception):
                    self._work_queue.put(None, timeout=1.0)

        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                logger.warning(f"Worker (pid={worker.pid}) didn't shutdown gracefully")
                worker.terminate()
                worker.join()

        # Clear queues
        for q in [self._work_queue, self._result_queue]:
            if q:
                # Clear the queue without getting stuck
                while not q.empty():
                    try:
                        q.get_nowait()
                    except (queue.Empty, FileNotFoundError):
                        # The queue might be empty due to race conditions
                        # or closed if the manager process is gone.
                        break

        # Cleanup
        self._workers.clear()
        self._work_queue = None
        self._result_queue = None
        self._shutdown_event = None
        if self._manager:
            # Check if manager process is still alive before shutting down
            if self._manager_process and self._manager_process.is_alive():
                self._manager.shutdown()
            self._manager = None
            self._manager_process = None

        logger.info("Workers shutdown complete")

    @staticmethod
    def _worker_loop(
        worker_id: int,
        pickled_dataset: bytes,
        work_queue: Queue,
        result_queue: Queue,
        shutdown_event: Event,
        pickled_collate_fn: bytes,
        pickled_init_fn: bytes | None,
        worker_stats: dict,
        timeout: float,
    ) -> None:
        """Main loop for worker processes.

        This is the main loop that runs in each of the worker processes
        and runs until the shutdown event is set.

        Args:
            worker_id: The ID of the worker process.
            pickled_dataset: The pickled dataset.
            work_queue: The queue to get work from.
            result_queue: The queue to put results in.
            shutdown_event: The event to signal shutdown.
            collate_fn: The function to collate batches.
            init_fn: The function to initialize the worker.
            worker_stats: The dictionary to store worker statistics.
            timeout: The timeout for getting work from the work queue.
        """
        # Ignore SIGINT in workers
        pid = os.getpid()
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Initialize worker
        try:
            if isinstance(pickled_dataset, bytes):
                # de-serialize the pre-pickled values (spawn context)
                dataset = pickle.load(io.BytesIO(pickled_dataset))
                init_fn = pickle.load(io.BytesIO(pickled_init_fn))
                collate_fn = pickle.load(io.BytesIO(pickled_collate_fn))
                del pickled_dataset, pickled_init_fn, pickled_collate_fn
                gc.collect()
            else:
                dataset = pickled_dataset
                init_fn = pickled_init_fn
                collate_fn = pickled_collate_fn

            # Set up worker info for PyTorch compatibility
            worker_info = WorkerInfo(
                id=worker_id,
                num_workers=len(worker_stats),
                seed=torch.initial_seed(),
                dataset=dataset,
            )
            torch.utils.data._utils.worker._worker_info = worker_info

            if init_fn is not None:
                init_fn(worker_id)

            stats = worker_stats[worker_id]
            stats.state = WorkerState.IDLE
            stats.last_update = time.time()

        except Exception as e:
            logger.error(f"Worker {worker_id} ({pid=}) initialization failed: {e}")
            stats.state = WorkerState.ERROR
            return

        # Main work loop
        while not shutdown_event.is_set():
            try:
                # Get work from the work queue with timeout
                batch_indices = work_queue.get(timeout=0.5)

                if batch_indices is None:  # Sentinel (indicates end of epoch)
                    break

                # Update state
                stats.state = WorkerState.LOADING
                stats.last_update = time.time()
                start_time = time.time()

                # Load batch
                batch = []
                for idx in batch_indices:
                    try:
                        logger.debug(f"Worker {worker_id} ({pid=}) loading index {idx}")
                        item = dataset[idx]
                        logger.debug(f"Worker {worker_id} ({pid=}) successfuly loaded index {idx}")
                        batch.append(item)
                    except Exception as e:
                        logger.error(f"Worker {worker_id} ({pid=}) failed to load index {idx}: {e}")
                        stats.errors += 1
                        raise e

                # Collate batch
                collated = collate_fn(batch)

                # Send result
                result_queue.put((worker_id, collated), timeout=timeout if timeout > 0 else None)

                # Update stats
                load_time = time.time() - start_time
                stats.items_processed += len(batch_indices)
                stats.total_load_time += load_time
                stats.state = WorkerState.IDLE
                stats.last_update = time.time()

            except queue.Empty:
                # No work available, continue
                continue

            except KeyboardInterrupt as e:
                # Propagate the KeyboardInterrupt to the main process
                raise e

            except Exception as e:
                # Send any other error to main process
                logger.error(f"Worker {worker_id} ({pid=}) encountered an error: {e}")
                stats.errors += 1
                stats.state = WorkerState.ERROR
                with contextlib.suppress(Exception):
                    result_queue.put((worker_id, e), timeout=1.0)

                # Don't crash the worker on single errors
                if stats.errors > 10:
                    logger.error(f"Worker {worker_id} ({pid=}) exceeded error threshold")
                    break

        # Cleanup
        stats.state = WorkerState.TERMINATED
        torch.utils.data._utils.worker._worker_info = None

    def __iter__(self) -> Iterator:
        """Iterate over batches."""
        if self.num_workers == 0:
            # Single-process loading
            for batch_indices in self.batch_sampler:
                batch = [self.dataset[idx] for idx in batch_indices]
                yield self._process_batch(self.collate_fn(batch))
        else:
            # Multi-process loading
            yield from self._multiprocess_iterator()

    def _fill_work_queue(self, batch_iter: Iterator) -> int:
        """Fills the work queue and returns the number of batches added."""
        count = 0
        # Fill the entire work queue at the start of the epoch.
        for _ in range(self.max_work_queue_size):
            batch_indices = next(batch_iter, None)
            if batch_indices is None:
                break
            self._work_queue.put(list(batch_indices))
            count += 1
        return count

    def _multiprocess_iterator(self) -> Iterator:
        """Multi-process iterator with work stealing."""
        assert self.num_workers > 1, "WorkStealingDataLoader requires at least two workers"

        batch_iter = iter(self.batch_sampler)

        if not self._workers:
            # One-time setup for the first epoch or non-persistent workers.
            ctx = mp.get_context(self.mp_context)
            self._manager = ctx.Manager()
            self._work_queue = ctx.Queue(maxsize=self.max_work_queue_size)
            self._result_queue = ctx.Queue(maxsize=self.max_result_queue_size)
            self._shutdown_event = ctx.Event()
            self._worker_stats = self._manager.dict()

            # Pre-fill queue before starting workers.
            self._fill_work_queue(batch_iter)
            self._init_workers(ctx)
        else:
            # For persistent workers on subsequent epochs, just fill the queue.
            self._fill_work_queue(batch_iter)

        total_batches = len(self.batch_sampler)
        batches_returned = 0
        try:
            # The main loop for yielding batches.
            # It runs as long as there are batches to be returned.
            while batches_returned < total_batches:
                try:
                    timeout = self.timeout if self.timeout > 0 else SAFETY_TIMEOUT
                    worker_id, result = self._result_queue.get(timeout=timeout)
                except queue.Empty:
                    self._check_worker_health()
                    raise RuntimeError(f"DataLoader timed out after {timeout}s")  # noqa: B904
                except Exception as e:
                    logger.error(f"Error getting result from queue: {e}")
                    raise

                if isinstance(result, Exception):
                    raise result

                yield self._process_batch(result)
                batches_returned += 1
        except GeneratorExit:
            # This is raised when the generator is closed.
            pass
        except KeyboardInterrupt:
            # This is raised when the user presses Ctrl+C.
            logger.info("Keyboard interrupt received, shutting down workers...")
            self._shutdown_workers()
            raise KeyboardInterrupt  # noqa: B904
        except Exception as e:
            logger.error(f"Error in _multiprocess_iterator: {e}")
            raise e
        finally:
            # This block is executed when the generator is exhausted or closed.
            # It ensures that the workers are shut down.
            if not self.persistent_workers:
                self._shutdown_workers()

    def _process_batch(self, batch: Any) -> Any:
        """Process batch (e.g., pin memory)."""
        if self.pin_memory:
            batch = pin_memory(batch)
        return batch

    def _check_worker_health(self) -> None:
        """Check if workers are healthy and log statistics."""
        if not self._worker_stats:
            return

        current_time = time.time()
        for worker_id, stats in self._worker_stats.items():
            if stats.state == WorkerState.ERROR:
                logger.warning(f"Worker {worker_id} in error state")
            elif current_time - stats.last_update > 60:
                logger.warning(f"Worker {worker_id} hasn't updated in 60s")

            # Log performance stats
            if stats.items_processed > 0:
                avg_time = stats.total_load_time / stats.items_processed
                logger.debug(
                    f"Worker {worker_id}: processed={stats.items_processed}, "
                    f"avg_time={avg_time:.3f}s, errors={stats.errors}"
                )

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.batch_sampler)

    def __del__(self):
        """Cleanup on deletion."""
        self._shutdown_workers()

    @contextmanager
    def worker_monitoring(self) -> Iterator[None]:
        """Context manager for monitoring worker performance."""
        start_time = time.time()
        initial_stats = {}

        try:
            if self._worker_stats:
                for worker_id, stats in self._worker_stats.items():
                    initial_stats[worker_id] = (
                        stats.items_processed,
                        stats.total_load_time,
                        stats.errors,
                    )
        except (BrokenPipeError, ConnectionError, EOFError, AttributeError):
            # Manager may be gone, or not initialized yet.
            pass

        yield

        # Log performance summary
        try:
            # Accessing self._worker_stats can fail if manager is shut down
            if self._worker_stats and self._manager and self._manager_process and self._manager_process.is_alive():
                duration = time.time() - start_time
                total_items = 0
                total_errors = 0

                current_stats = self._worker_stats.copy()
                for worker_id, stats in current_stats.items():
                    if worker_id in initial_stats:
                        items_processed = stats.items_processed - initial_stats[worker_id][0]
                        errors = stats.errors - initial_stats[worker_id][2]

                        total_items += items_processed
                        total_errors += errors

                        if items_processed > 0:
                            throughput = items_processed / duration
                            logger.info(
                                f"Worker {worker_id}: {items_processed} items, "
                                f"{throughput:.1f} items/s, {errors} errors"
                            )

                if total_items > 0:
                    logger.info(
                        f"Total: {total_items} items in {duration:.1f}s "
                        f"({total_items/duration:.1f} items/s), {total_errors} errors"
                    )
        except (BrokenPipeError, ConnectionError, EOFError, AttributeError):
            # Workers have been shut down, skip logging detailed stats
            duration = time.time() - start_time
            logger.info(f"Workers completed in {duration:.1f}s (stats unavailable after shutdown)")
