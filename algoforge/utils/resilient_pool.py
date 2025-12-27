"""Resilient process pool that auto-recovers from worker crashes."""

import asyncio
import threading
from concurrent.futures import ProcessPoolExecutor, BrokenExecutor
from typing import Callable, Any, TypeVar

T = TypeVar('T')


class ResilientProcessPool:
    """A ProcessPoolExecutor wrapper that automatically recreates the pool when workers crash.

    When a child process terminates abruptly (OOM, segfault, etc.), the ProcessPoolExecutor
    becomes permanently broken and all subsequent tasks fail with BrokenProcessPool.

    This wrapper catches those failures and transparently recreates the pool, allowing
    the system to continue operating.

    Usage:
        pool = ResilientProcessPool(max_workers=8, max_tasks_per_child=5)

        # In async context:
        result = await pool.run(loop, some_function, arg1, arg2)

        # Cleanup:
        pool.shutdown()

    Thread-safe: Can be used from multiple async tasks concurrently.
    """

    def __init__(
        self,
        max_workers: int = None,
        max_tasks_per_child: int = None,
        mp_context=None,
        initializer: Callable = None,
        initargs: tuple = (),
    ):
        """Initialize the resilient process pool.

        Args:
            max_workers: Maximum number of worker processes
            max_tasks_per_child: Number of tasks a worker can execute before being replaced
            mp_context: Multiprocessing context (e.g., 'spawn', 'fork')
            initializer: Callable to run in each worker on startup
            initargs: Arguments to pass to initializer
        """
        self._max_workers = max_workers
        self._max_tasks_per_child = max_tasks_per_child
        self._mp_context = mp_context
        self._initializer = initializer
        self._initargs = initargs

        self._lock = threading.Lock()
        self._executor: ProcessPoolExecutor | None = None
        self._recreation_count = 0
        self._is_shutdown = False

        # Create initial executor
        self._create_executor()

    def _create_executor(self) -> None:
        """Create a new ProcessPoolExecutor instance."""
        kwargs = {}
        if self._max_workers is not None:
            kwargs['max_workers'] = self._max_workers
        if self._max_tasks_per_child is not None:
            kwargs['max_tasks_per_child'] = self._max_tasks_per_child
        if self._mp_context is not None:
            kwargs['mp_context'] = self._mp_context
        if self._initializer is not None:
            kwargs['initializer'] = self._initializer
            kwargs['initargs'] = self._initargs

        self._executor = ProcessPoolExecutor(**kwargs)

    def _recreate_executor(self) -> None:
        """Shutdown broken executor and create a new one."""
        with self._lock:
            if self._is_shutdown:
                return

            # Shutdown old executor (don't wait, it's broken anyway)
            if self._executor is not None:
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass  # Ignore errors during shutdown of broken executor

            # Create new executor
            self._create_executor()
            self._recreation_count += 1
            print(f"  [ResilientPool] Recreated executor (recreation #{self._recreation_count})", flush=True)

    async def run(
        self,
        loop: asyncio.AbstractEventLoop,
        fn: Callable[..., T],
        *args: Any,
        max_retries: int = 1,
    ) -> T:
        """Run a function in the process pool with automatic recovery.

        Args:
            loop: The asyncio event loop
            fn: Function to execute (must be picklable)
            *args: Arguments to pass to fn
            max_retries: Number of retries after pool recreation (default: 1)

        Returns:
            The result of fn(*args)

        Raises:
            BrokenExecutor: If the pool crashes and max_retries is exceeded
            Exception: Any exception raised by fn
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                with self._lock:
                    if self._is_shutdown:
                        raise RuntimeError("Pool is shutdown")
                    executor = self._executor

                return await loop.run_in_executor(executor, fn, *args)

            except BrokenExecutor as e:
                last_error = e
                if attempt < max_retries:
                    print(f"  [ResilientPool] Pool broken, recreating (attempt {attempt + 1}/{max_retries + 1})", flush=True)
                    self._recreate_executor()
                else:
                    # Re-raise on final attempt
                    raise

        # Should not reach here, but just in case
        raise last_error

    def submit(self, fn: Callable[..., T], *args: Any) -> 'Future[T]':
        """Submit a task directly to the executor (synchronous interface).

        Note: This doesn't auto-recover. Use `run()` for async recovery.

        Returns:
            A Future representing the pending result
        """
        with self._lock:
            if self._is_shutdown:
                raise RuntimeError("Pool is shutdown")
            return self._executor.submit(fn, *args)

    @property
    def recreation_count(self) -> int:
        """Number of times the pool has been recreated due to crashes."""
        return self._recreation_count

    @property
    def executor(self) -> ProcessPoolExecutor:
        """Access the underlying executor (for run_in_executor compatibility)."""
        with self._lock:
            return self._executor

    def shutdown(self, wait: bool = False, cancel_futures: bool = True) -> None:
        """Shutdown the process pool.

        Args:
            wait: Whether to wait for pending tasks to complete
            cancel_futures: Whether to cancel pending futures
        """
        with self._lock:
            self._is_shutdown = True
            if self._executor is not None:
                try:
                    self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
                except Exception:
                    pass
                self._executor = None

    def __enter__(self) -> 'ResilientProcessPool':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
