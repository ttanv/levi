"""Process pool with hard timeout enforcement via process termination."""

import asyncio
import atexit
import multiprocessing as mp
import signal
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def _worker_fn(fn: Callable, args: tuple, result_queue: mp.Queue) -> None:
    try:
        result = fn(*args)
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", f"{type(e).__name__}: {e}"))


class ResilientProcessPool:
    _instances: list["ResilientProcessPool"] = []

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_workers)
        self._active: dict[int, mp.Process] = {}
        self._lock = threading.Lock()
        self._shutdown = False
        self._ctx = mp.get_context("spawn")
        ResilientProcessPool._instances.append(self)

    async def run(self, fn: Callable[..., T], *args: Any, timeout: float) -> T:
        if self._shutdown:
            raise RuntimeError("Pool is shutdown")
        async with self._semaphore:
            return await self._execute(fn, args, timeout)

    async def _execute(self, fn: Callable[..., T], args: tuple, timeout: float) -> T:
        try:
            result_queue = self._ctx.Queue(maxsize=1)
            proc = self._ctx.Process(target=_worker_fn, args=(fn, args, result_queue), daemon=True)
            proc.start()
        except (OSError, PermissionError):
            return await self._execute_inline(fn, args, timeout)

        with self._lock:
            self._active[proc.pid] = proc

        try:
            start = time.monotonic()
            while time.monotonic() - start < timeout:
                if not proc.is_alive():
                    break
                await asyncio.sleep(0.1)

            if proc.is_alive():
                self._terminate_process(proc)
                raise TimeoutError(f"Process exceeded {timeout}s timeout")

            try:
                status, value = result_queue.get(timeout=2.0)
            except Exception:
                exitcode = proc.exitcode
                if exitcode is not None and exitcode < 0:
                    try:
                        signame = signal.Signals(-exitcode).name
                        raise RuntimeError(f"Process killed by {signame}")
                    except ValueError:
                        raise RuntimeError(f"Process killed by signal {-exitcode}")
                raise RuntimeError(f"Process exited with code {exitcode}, no result")

            if status == "error":
                raise RuntimeError(value)
            return value

        finally:
            with self._lock:
                self._active.pop(proc.pid, None)
            self._terminate_process(proc)
            try:
                result_queue.close()
                result_queue.join_thread()
            except Exception:
                pass

    async def _execute_inline(self, fn: Callable[..., T], args: tuple, timeout: float) -> T:
        try:
            return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(f"Inline evaluation exceeded {timeout}s timeout") from exc

    def _terminate_process(self, proc: mp.Process) -> None:
        if not proc.is_alive():
            return
        try:
            proc.terminate()
            proc.join(timeout=2.0)
        except Exception:
            pass
        if proc.is_alive():
            try:
                proc.kill()
                proc.join(timeout=2.0)
            except Exception:
                pass

    def shutdown(self) -> None:
        self._shutdown = True
        with self._lock:
            for proc in list(self._active.values()):
                self._terminate_process(proc)
            self._active.clear()
        try:
            ResilientProcessPool._instances.remove(self)
        except ValueError:
            pass

    def __enter__(self) -> "ResilientProcessPool":
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()

    @classmethod
    def _cleanup_all(cls) -> None:
        for pool in list(cls._instances):
            try:
                pool.shutdown()
            except Exception:
                pass


atexit.register(ResilientProcessPool._cleanup_all)


def _handle_signals(signum, frame):
    ResilientProcessPool._cleanup_all()
    import sys

    sys.exit(128 + signum)


if mp.current_process().name == "MainProcess":
    try:
        signal.signal(signal.SIGTERM, _handle_signals)
    except Exception:
        pass
